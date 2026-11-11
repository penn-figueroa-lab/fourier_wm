"""JIT batched projected damped Gauss-Newton (the former SQP planner).

Only imports NumPy/JAX: use this module in a separate process/environment when
Torch and JAX CUDA libraries conflict. Export parameters using numpy_parameters.
"""

import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_default_matmul_precision", "highest")


def stack_parameters(parameters):
    """One dict or a list of independently trained heads -> leading head axis."""
    heads = [parameters] if isinstance(parameters, dict) else list(parameters)
    if not heads:
        raise ValueError("At least one parameter dictionary is required")
    keys = set(heads[0])
    allowed = [
        {"autonomous_matrix", "B"},
        {"autonomous_matrix", "B", "bilinear_control"},
        {"autonomous_matrix", "B", "omega_f", "omega_g", "fourier_f", "fourier_g"},
    ]
    if keys not in allowed or any(set(head) != keys for head in heads):
        raise ValueError("Unexpected parameter keys")
    return {key: jnp.asarray(np.stack([p[key] for p in heads]), dtype=jnp.float32) for key in keys}


def make_sqp(
    horizon=25,
    action_dim=2,
    iterations=10,
    smoothness=0.1,
    magnitude=0.0,
    damping=0.01,
    bound=3.0,
    max_step=1.0,
    lower=None,
    upper=None,
):
    """Return JIT solve(params, z0[B,D], goal[B,D], initial[B,H,m]).

    params has a leading ensemble axis; each member rolls out independently.
    Members share encoder coordinates. Cost is mean terminal MSE + mean input
    slew penalty (+ optional magnitude). No latent/path/learned cost.
    Uses a damped GN Hessian, not the exact nonlinear Hessian, with a five-point
    projected line search. This is NOT an exact active-set box-QP solver or a
    convergence/safety guarantee. The zero-step candidate ensures no increase
    in the learned-model objective for finite arithmetic and feasible starts.
    """
    if min(horizon, action_dim) < 1 or iterations < 0:
        raise ValueError("Require positive dimensions and nonnegative iterations")
    if any(not math.isfinite(x) for x in (smoothness, magnitude, damping, bound, max_step)):
        raise ValueError("Options must be finite")
    if min(smoothness, magnitude) < 0 or min(damping, bound, max_step) <= 0:
        raise ValueError("Invalid regularization or action limits")
    n = horizon * action_dim
    lower = np.full(action_dim, -bound) if lower is None else np.asarray(lower)
    upper = np.full(action_dim, bound) if upper is None else np.asarray(upper)
    if (
        lower.shape != (action_dim,)
        or upper.shape != (action_dim,)
        or not np.isfinite([lower, upper]).all()
        or (lower >= upper).any()
    ):
        raise ValueError("Invalid per-channel command bounds")
    lower_flat, upper_flat = jnp.tile(lower, horizon), jnp.tile(upper, horizon)
    difference = jnp.eye(n)[action_dim:] - jnp.eye(n)[:-action_dim]
    slew_denominator = max((horizon - 1) * action_dim, 1)

    def one(p, state, controls):
        def step(z, u):
            if "omega_f" not in p:
                control = p["B"]
                if "bilinear_control" in p:
                    control = control + jnp.einsum("dmj,j->dm", p["bilinear_control"], z)
                return z + p["autonomous_matrix"] @ z + control @ u, None
            pf, pg = p["omega_f"] @ z, p["omega_g"] @ z
            features_f = jnp.concatenate((jnp.sin(pf), jnp.cos(pf)))
            features_g = jnp.concatenate((jnp.sin(pg), jnp.cos(pg)))
            control = p["B"] + (p["fourier_g"] @ features_g).reshape(z.shape[-1], action_dim)
            return z + p["autonomous_matrix"] @ z + p["fourier_f"] @ features_f + control @ u, None

        return jax.lax.scan(step, state, controls)[0]

    heads = jax.vmap(jax.vmap(one, in_axes=(None, 0, 0)), in_axes=(0, None, None))
    derivatives = jax.vmap(
        jax.vmap(jax.jacfwd(one, argnums=2), in_axes=(None, 0, 0)), in_axes=(0, None, None)
    )

    def endpoint(p, z, u):
        value = heads(p, z, u)
        return value.transpose(1, 0, 2).reshape(u.shape[0], -1)

    def cost(p, z, goal, u):
        target = jnp.tile(goal, (1, p["B"].shape[0]))
        terminal = jnp.mean((endpoint(p, z, u) - target) ** 2, -1)
        slew = jnp.sum(jnp.diff(u, axis=1) ** 2, (1, 2)) / slew_denominator
        return terminal + smoothness * slew + magnitude * jnp.mean(u**2, (1, 2))

    def solve(p, z, goal, initial):
        if (
            z.ndim != 2
            or goal.shape != z.shape
            or initial.shape != (z.shape[0], horizon, action_dim)
        ):
            raise ValueError("Expected z/goal[B,D] and initial[B,H,m]")
        batch, width = z.shape
        d = width * p["B"].shape[0]
        target = jnp.tile(goal, (1, p["B"].shape[0]))
        reg = (2 * smoothness * d / slew_denominator) * (difference.T @ difference)
        reg = reg + 2 * magnitude * d / n * jnp.eye(n)

        def iteration(_, u):
            flat = u.reshape(batch, n)
            residual = endpoint(p, z, u) - target
            jacobian = derivatives(p, z, u).transpose(1, 0, 2, 3, 4).reshape(batch, d, n)
            gradient = 2 * jnp.einsum("bdi,bd->bi", jacobian, residual) + flat @ reg
            normal = 2 * jacobian.transpose(0, 2, 1) @ jacobian + reg + damping * jnp.eye(n)
            direction = jnp.linalg.solve(normal, gradient[..., None])[..., 0]
            direction *= jnp.minimum(
                1.0, max_step / jnp.maximum(jnp.max(jnp.abs(direction), -1, keepdims=True), 1e-8)
            )
            alphas = jnp.array([1.0, 0.5, 0.25, 0.125, 0.0])
            candidates = jnp.clip(
                flat[:, None] - alphas[None, :, None] * direction[:, None], lower_flat, upper_flat
            )
            candidates = candidates.reshape(-1, horizon, action_dim)
            losses = cost(p, jnp.repeat(z, 5, 0), jnp.repeat(goal, 5, 0), candidates).reshape(
                batch, 5
            )
            return candidates.reshape(batch, 5, horizon, action_dim)[
                jnp.arange(batch), jnp.argmin(losses, -1)
            ]

        return jax.lax.fori_loop(
            0, iterations, iteration, jnp.clip(initial, jnp.asarray(lower), jnp.asarray(upper))
        )

    return jax.jit(solve), jax.jit(cost)


def shift_warm_start(controls, executed_steps):
    """Shift [B,H,m], repeat final input to fill tail; use recent observation."""
    if not 1 <= executed_steps <= controls.shape[1]:
        raise ValueError("executed_steps must lie in [1,H]")
    return jnp.concatenate(
        (controls[:, executed_steps:], jnp.repeat(controls[:, -1:], executed_steps, axis=1)), axis=1
    )


# SWM protocol adapter; the supplied JAX kernels above are unchanged.
class SWMSQPSolver:
    def __init__(
        self,
        cost,
        iterations=10,
        smoothness=0.1,
        magnitude=0.0,
        damping=0.01,
        bound=3.0,
        max_step=1.0,
        device="cuda",
        stats_dataset=None,
        metadata=None,
        starts=1,
        seed=42,
        initialization="gaussian",
        initialization_amplitude=0.8,
    ):
        self.cost = cost
        self.device = device
        self.options = dict(
            iterations=iterations,
            smoothness=smoothness,
            magnitude=magnitude,
            damping=damping,
            bound=bound,
            max_step=max_step,
        )
        self.stats_dataset = stats_dataset
        self.metadata = metadata
        self.starts = int(starts)
        if self.starts < 1:
            raise ValueError("starts must be positive")
        if initialization not in {"gaussian", "compass"}:
            raise ValueError("Choose gaussian or compass initialization")
        if not math.isfinite(initialization_amplitude) or not 0 < initialization_amplitude <= 1:
            raise ValueError("Initialization amplitude must be a fraction in (0, 1]")
        self.initialization = initialization
        self.initialization_amplitude = initialization_amplitude
        self.rng = np.random.default_rng(seed)
        self.seed = int(seed)
        self._configured = False
        self.compilation_seconds = 0.0
        self._compiled = {}

    def configure(self, *, action_space, n_envs, config):
        from sklearn.preprocessing import StandardScaler
        from stable_worldmodel.data import load_dataset

        self._config, self._n_envs = config, n_envs
        self._primitive_dim = int(np.prod(action_space.shape[1:]))
        model = self.cost.model
        dynamics = getattr(model.predictor, "dynamics", model.predictor)
        if self._primitive_dim != dynamics.action_dim:
            raise ValueError("Environment/model primitive action dimensions differ")
        if config.action_block != getattr(
            model, "action_block", getattr(model.predictor, "action_block", None)
        ):
            raise ValueError("Training/planning action blocks differ")
        self._params = stack_parameters(dynamics.numpy_parameters())
        metadata = self.metadata
        if isinstance(metadata, (str, Path)):
            import json

            metadata = json.loads(Path(metadata).read_text())
        if metadata is not None:
            self._mean = np.asarray(metadata["action_mean"])
            self._scale = np.asarray(metadata["action_scale"])
        elif hasattr(model, "action_mean"):
            self._mean = model.action_mean.detach().float().cpu().numpy()
            self._scale = model.action_scale.detach().float().cpu().numpy()
        else:
            raise ValueError("Current Fourier checkpoints require their training_metadata.json")
        from .actions import ActionNormalizer, ActionAdapter

        ActionNormalizer(self._mean, self._scale)
        limits = {}
        self._action_adapter = None
        if metadata and metadata.get("action_spec"):
            adapter = ActionAdapter(metadata["action_spec"])
            self._action_adapter = adapter
            if not adapter.state_dependent:
                lo, hi = adapter.command_bounds()
                limits = {
                    "lower": np.maximum((lo - self._mean) / self._scale, -self.options["bound"]),
                    "upper": np.minimum((hi - self._mean) / self._scale, self.options["bound"]),
                }
        if self.initialization == "compass":
            adapter = self._action_adapter
            if (
                self._primitive_dim != 2
                or self.starts != 8
                or adapter is None
                or adapter.state_dependent
                or any(k not in {"delta", "velocity"} for k in adapter.spec.kinds)
                or not np.allclose(adapter.spec.low, -np.asarray(adapter.spec.high))
            ):
                raise ValueError(
                    "Compass initialization requires eight starts and two symmetric, state-independent velocity/displacement channels with checkpoint action metadata"
                )
        self._solve, self._cost = make_sqp(
            horizon=config.horizon * config.action_block,
            action_dim=self._primitive_dim,
            **self.options,
            **limits,
        )
        self._compiled.clear()
        if self.stats_dataset:
            ds = load_dataset(self.stats_dataset, keys_to_load=["action"], keys_to_cache=["action"])
            values = ds.get_col_data("action")
            values = values[~np.isnan(values).any(axis=1)]
            scaler = StandardScaler().fit(values)
            self._eval_mean, self._eval_scale = scaler.mean_, scaler.scale_
        else:
            self._eval_mean, self._eval_scale = self._mean, self._scale
        self._configured = True

    def initial_candidates(self, initial, warm=False):
        """Initial guesses only: every primitive input remains free in the solve.

        Compass directions use native command bounds, then the checkpoint's
        command conversion and normalization. No observation/goal state enters
        initialization. After the first plan, slot zero retains the warm start.
        """
        if self.initialization == "compass":
            angles = np.arange(8) * np.pi / 4
            native = self.initialization_amplitude * np.stack([np.cos(angles), np.sin(angles)], -1)
            native *= np.asarray(self._action_adapter.spec.high)
            commands = (self._action_adapter.to_model(native) - self._mean) / self._scale
            candidates = (
                np.broadcast_to(commands[None, :, None], (len(initial), 8, initial.shape[1], 2))
                .copy()
                .astype(np.float32)
            )
            if warm:
                candidates[:, 0] = initial
            return candidates
        candidates = np.repeat(np.asarray(initial)[:, None], self.starts, axis=1)
        if self.starts > 1:
            candidates[:, 1] = 0
        if self.starts > 2:
            candidates[:, 2:] = self.rng.normal(size=candidates[:, 2:].shape)
        return candidates

    @property
    def n_envs(self):
        return self._n_envs

    @property
    def action_dim(self):
        return self._primitive_dim * self._config.action_block

    @property
    def horizon(self):
        return self._config.horizon

    def __call__(self, *args, **kwargs):
        return self.solve(*args, **kwargs)

    def solve(self, info_dict, init_action=None):
        import torch

        if not self._configured:
            raise RuntimeError("Call configure first")
        model = self.cost.model
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype
        local = {
            k: v.to(device=device, dtype=dtype if v.is_floating_point() else None)
            if torch.is_tensor(v)
            else v
            for k, v in info_dict.items()
        }
        with torch.no_grad():
            current = model.encode(dict(local))["emb"][:, -1]
            expanded = {k: v.unsqueeze(1) if torch.is_tensor(v) else v for k, v in local.items()}
            goal = self.cost.encode_goal(model, expanded)[:, -1]
        z = jnp.asarray(current.float().cpu().numpy())
        target = jnp.asarray(goal.float().cpu().numpy())
        batch = z.shape[0]
        primitive_horizon = self.horizon * self._config.action_block
        if init_action is None or init_action.shape[1] == 0:
            initial = np.zeros((batch, primitive_horizon, self._primitive_dim), dtype=np.float32)
        else:
            prefix = (
                init_action.detach().float().cpu().numpy().reshape(batch, -1, self._primitive_dim)
            )
            prefix = (prefix * self._eval_scale + self._eval_mean - self._mean) / self._scale
            remaining = primitive_horizon - prefix.shape[1]
            if remaining < 0:
                raise ValueError("Warm start exceeds horizon")
            initial = np.concatenate([prefix, np.repeat(prefix[:, -1:], remaining, axis=1)], axis=1)
        initial = jnp.asarray(initial, dtype=jnp.float32)
        candidates = self.initial_candidates(
            np.asarray(initial), warm=init_action is not None and init_action.shape[1] > 0
        )
        zz, gg = jnp.repeat(z, self.starts, 0), jnp.repeat(target, self.starts, 0)
        initial_candidates = jnp.asarray(
            candidates.reshape(-1, primitive_horizon, self._primitive_dim)
        )
        key = (zz.shape, initial_candidates.shape)
        if key not in self._compiled:
            import time

            begin = time.perf_counter()
            self._compiled[key] = (
                self._solve.lower(self._params, zz, gg, initial_candidates).compile(),
                self._cost.lower(self._params, zz, gg, initial_candidates).compile(),
            )
            self.compilation_seconds += time.perf_counter() - begin
        solve, cost = self._compiled[key]
        optimized = solve(self._params, zz, gg, initial_candidates)
        all_costs = np.asarray(cost(self._params, zz, gg, optimized)).reshape(batch, self.starts)
        best = np.argmin(all_costs, axis=1)
        final_cost = all_costs[np.arange(batch), best]
        controls = np.asarray(optimized).reshape(
            batch, self.starts, primitive_horizon, self._primitive_dim
        )[np.arange(batch), best]
        if not np.isfinite(controls).all() or not np.isfinite(final_cost).all():
            raise ValueError("Non-finite SQP output")
        # Convert model-normalized controls back to stock policy's scaler space.
        result = (controls * self._scale + self._mean - self._eval_mean) / self._eval_scale
        actions = torch.from_numpy(np.array(result, dtype=np.float32)).reshape(
            batch, self.horizon, self.action_dim
        )
        return {"actions": actions, "costs": final_cost.tolist()}
