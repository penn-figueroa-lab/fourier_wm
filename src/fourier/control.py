"""Primitive-step MPC using SWM solvers and optional Fourier TV-LQR feedback."""

from collections import deque
from copy import copy
from dataclasses import replace
import time

import numpy as np
import torch
from stable_worldmodel.policy import BasePolicy


def dynamics_of(model):
    dynamics = getattr(model.predictor, "dynamics", model.predictor)
    if not hasattr(dynamics, "terminal_with_action_jacobian"):
        raise ValueError("TV-LQR requires a primitive Fourier dynamics checkpoint")
    return dynamics


@torch.no_grad()
def feedback_plan(dynamics, initial, actions, q=1.0, r=1.0, terminal=1.0):
    """Exact local Jacobians and finite-horizon Riccati recursion in latent space."""
    if min(q, r, terminal) <= 0:
        raise ValueError("TV-LQR weights must be positive")
    states = torch.cat((initial[None], dynamics.rollout(initial, actions)), 0)
    # Batched Jacobians avoid a backward call for every physical step.
    jac = torch.func.vmap(torch.func.jacrev(dynamics, argnums=(0, 1)))
    aa, bb = jac(states[:-1], actions)
    p = torch.eye(initial.numel(), device=initial.device, dtype=initial.dtype) * terminal
    eye_z = torch.eye(initial.numel(), device=initial.device, dtype=initial.dtype)
    eye_u = torch.eye(actions.shape[-1], device=initial.device, dtype=initial.dtype)
    gains = []
    for a, b in zip(aa.flip(0), bb.flip(0)):
        k = torch.linalg.solve(r * eye_u + b.T @ p @ b, b.T @ p @ a)
        p = q * eye_z + a.T @ p @ (a - b @ k)
        p = (p + p.T) * 0.5
        gains.append(k)
    return states, torch.stack(gains[::-1])


class PrimitiveMPCPolicy(BasePolicy):
    """Histories contain actual observations and actually executed commands only.

    Replan cadence is independent of the checkpoint's action-block size.
    Per-env plans are reset independently, including on a new evaluation run.
    """

    def __init__(
        self,
        solver,
        config,
        *,
        replan_steps,
        process=None,
        transform=None,
        adapter=None,
        feedback=False,
        feedback_q=1.0,
        feedback_r=1.0,
        clip_actions=True,
        horizon_schedule="fixed",
    ):
        super().__init__()
        self.solver, self.cfg = solver, config
        self.process, self.transform = process or {}, transform or {}
        self.adapter, self.feedback = adapter, feedback
        self.clip_actions = clip_actions
        self.feedback_q, self.feedback_r = feedback_q, feedback_r
        self.replan_steps = int(replan_steps)
        if horizon_schedule not in {"fixed", "deadline"}:
            raise ValueError("Choose fixed or deadline horizon scheduling")
        if horizon_schedule == "deadline" and self.replan_steps % config.action_block:
            raise ValueError(
                "Deadline replanning requires complete action blocks; use five-step replanning and per-observation TV-LQR, or a fixed horizon for single-step replanning"
            )
        self.horizon_schedule = horizon_schedule
        self.replan_horizons = []
        if not 1 <= self.replan_steps <= config.horizon * config.action_block:
            raise ValueError("Replan interval must be within the primitive planning horizon")
        self.timings = {
            "plan_seconds": [],
            "compilation_seconds": [],
            "feedback_seconds": [],
            "action_seconds": [],
        }
        self.clipped_actions = 0

    def set_env(self, env):
        self.env = env
        self.solver.configure(action_space=env.action_space, n_envs=env.num_envs, config=self.cfg)
        self._horizon_solvers = {self.cfg.horizon: self.solver}
        self.reset()

    def reset(self):
        env = self.env
        self._next_cache_id = getattr(self, "_next_cache_id", 0) + env.num_envs
        self.cache_ids = np.arange(self._next_cache_id, self._next_cache_id + env.num_envs)
        self.observation_steps = np.zeros(env.num_envs, dtype=int)
        predictor = getattr(self.solver.cost.model, "predictor", None)
        self.history_stride = getattr(
            predictor, "history_stride_physical_steps", self.cfg.action_block
        )
        self.capacity = (self.cfg.history_len - 1) * self.history_stride + 1
        self.observations = [deque(maxlen=self.capacity) for _ in range(env.num_envs)]
        self.commands = [deque(maxlen=self.capacity) for _ in range(env.num_envs)]
        self.plans = [None] * env.num_envs
        self.feedback_data = [None] * env.num_envs
        self.positions = np.zeros(env.num_envs, dtype=int)

    def _reset(self, i):
        self._next_cache_id = max(self._next_cache_id, int(self.cache_ids.max())) + 1
        self.cache_ids[i] = self._next_cache_id
        self.observation_steps[i] = 0
        self.observations[i].clear()
        self.commands[i].clear()
        self.plans[i] = self.feedback_data[i] = None
        self.positions[i] = 0

    def _info(self, i, prepared):
        current = {k: v[i : i + 1] for k, v in prepared.items() if torch.is_tensor(v)}
        # Upstream DINO caches encoded contexts using these causal identifiers.
        current["id"] = torch.tensor([[self.cache_ids[i]]])
        current["step_idx"] = torch.tensor([[self.observation_steps[i]]])
        history = list(self.observations[i])
        indices = list(range(len(history) - 1, -1, -self.history_stride))[: self.cfg.history_len][
            ::-1
        ]
        model = self.solver.cost.model
        keys = (
            *getattr(model, "observation_keys", ("pixels",)),
            *[k for k in getattr(model, "extra_encoders", {}) if k != "action"],
        )
        for k in keys:
            current[k] = torch.cat([history[j][k] for j in indices], dim=1)
        if len(indices) > 1:
            # One command leaves each recorded observation, except the newest.
            commands = list(self.commands[i])
            count = (len(indices) - 1) * self.history_stride
            current["action_history"] = torch.from_numpy(np.stack(commands[-count:])).reshape(
                1, len(indices) - 1, -1
            )
        return current

    def get_action(self, info_dict, **kwargs):
        started = time.perf_counter()
        flush = np.asarray(info_dict.get("_needs_flush", np.zeros(self.env.num_envs))).reshape(-1)
        dead = np.asarray(info_dict.get("terminated", np.zeros(self.env.num_envs))).reshape(-1)
        # We store executed model commands ourselves: an absolute native target
        # from the previous step cannot be reconverted using the new position.
        clean = {k: v for k, v in info_dict.items() if k != "_needs_flush"}
        prepared = self._prepare_info(clean)
        result = np.zeros(self.env.action_space.shape, dtype=np.float32)
        model = self.solver.cost.model
        device = next(model.parameters()).device
        for i in range(self.env.num_envs):
            if flush[i]:
                self._reset(i)
            if dead[i]:
                continue
            history_keys = {
                *getattr(model, "observation_keys", ("pixels",)),
                *[k for k in getattr(model, "extra_encoders", {}) if k != "action"],
            }
            self.observations[i].append(
                {k: v[i : i + 1].clone() for k, v in prepared.items() if k in history_keys}
            )
            local = self._info(i, prepared)
            pos = self.positions[i]
            if self.plans[i] is None or pos >= self.replan_steps:
                horizon = self.cfg.horizon
                if self.horizon_schedule == "deadline":
                    horizon -= (self.observation_steps[i] // self.cfg.action_block) % horizon
                horizon = int(horizon)
                if horizon not in self._horizon_solvers:
                    # Reuse the model with independent seeded streams and
                    # compiled solver kernels for each shrinking horizon.
                    solver = copy(self.solver)
                    if hasattr(solver, "rng") and hasattr(solver, "seed"):
                        solver.rng = np.random.default_rng(solver.seed)
                    if hasattr(solver, "torch_gen"):
                        solver.torch_gen = torch.Generator(
                            device=solver.torch_gen.device
                        ).manual_seed(solver.torch_gen.initial_seed())
                    if hasattr(solver, "_compiled"):
                        solver._compiled = {}
                        solver.compilation_seconds = 0.0
                    solver.configure(
                        action_space=self.env.action_space,
                        n_envs=self.env.num_envs,
                        config=replace(self.cfg, horizon=horizon),
                    )
                    self._horizon_solvers[horizon] = solver
                solver = self._horizon_solvers[horizon]
                warm = None
                if self.plans[i] is not None and self.cfg.warm_start:
                    old = self.plans[i]
                    # Match upstream CEM's prepare_init_action: retain the
                    # unexecuted prefix and zero-pad newly exposed timesteps
                    # in solver coordinates. Repeating the terminal command
                    # also wrongly warm-started a fully consumed plan.
                    warm = np.zeros(
                        (horizon * self.cfg.action_block, old.shape[-1]), dtype=old.dtype
                    )
                    remaining = old[pos : pos + len(warm)]
                    warm[: len(remaining)] = remaining
                    warm = torch.from_numpy(warm).reshape(1, horizon, -1)
                t = time.perf_counter()
                compiled_before = getattr(solver, "compilation_seconds", 0.0)
                output = solver(local, init_action=warm)
                self.replan_horizons.append(horizon * self.cfg.action_block)
                plan = (
                    output["actions"]
                    .detach()
                    .cpu()
                    .numpy()
                    .reshape(-1, self.env.single_action_space.shape[0])
                )
                if not np.isfinite(plan).all():
                    raise ValueError("Planner returned nonfinite commands")
                self.plans[i], self.positions[i], pos = plan, 0, 0
                if self.feedback:
                    current = {k: v.to(device) for k, v in local.items() if torch.is_tensor(v)}
                    with torch.no_grad():
                        z = model.encode(current)["emb"][0, -1]
                    self.feedback_data[i] = feedback_plan(
                        dynamics_of(model),
                        z,
                        torch.tensor(plan, device=device),
                        self.feedback_q,
                        self.feedback_r,
                    )
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                compiled = getattr(solver, "compilation_seconds", 0.0) - compiled_before
                self.timings["compilation_seconds"].append(compiled)
                self.timings["plan_seconds"].append(time.perf_counter() - t - compiled)
            command = self.plans[i][pos].copy()
            if self.feedback:
                t = time.perf_counter()
                with torch.no_grad():
                    z = model.encode(
                        {k: v.to(device) for k, v in local.items() if torch.is_tensor(v)}
                    )["emb"][0, -1]
                    states, gains = self.feedback_data[i]
                    command -= (gains[pos] @ (z - states[pos])).cpu().numpy()
                self.timings["feedback_seconds"].append(time.perf_counter() - t)
            physical = self.process["action"].inverse_transform(command[None])[0]
            q = None
            if self.adapter and self.adapter.state_dependent:
                q = np.asarray(info_dict[self.adapter.spec.position_key])[i, -1]
            native = self.adapter.to_native(physical, q) if self.adapter else physical
            clipped = (
                np.clip(native, self.env.single_action_space.low, self.env.single_action_space.high)
                if self.clip_actions
                else native
            )
            self.clipped_actions += int(np.any(clipped != native))
            actual = self.adapter.to_model(clipped, q) if self.adapter else clipped
            self.commands[i].append(self.process["action"].transform(actual[None])[0])
            result[i] = clipped
            self.positions[i] += 1
            self.observation_steps[i] += 1
        self.timings["action_seconds"].append(time.perf_counter() - started)
        return result
