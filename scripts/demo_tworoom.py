#!/usr/bin/env python3
"""Run shared Fourier weights on one native Two-Room task without a dataset."""

import argparse
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import imageio.v3 as iio
import numpy as np
import torch
from omegaconf import OmegaConf

from fourier.checkpoints import inspect_checkpoint, load_pretrained
from fourier.configuration import compose
from fourier.environments import make_env, success
from fourier.evaluation import checkpoint_metadata, make_policy
from fourier.runtime import RunDirectory, atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="New output directory")
    parser.add_argument("--planner", choices=("cem", "sqp"), default="cem")
    parser.add_argument("--feedback", action="store_true", help="Enable TV-LQR with sqp")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.steps < 1 or (args.feedback and args.planner != "sqp"):
        parser.error("Use positive --steps; --feedback requires --planner sqp")
    torch.set_num_threads(2)
    cfg = compose(
        overrides=["environment=tworoom", "protocol=closed_loop", "planner=" + args.planner]
    )
    cfg.machine.device = args.device
    cfg.seed = args.seed
    cfg.protocol.feedback = args.feedback
    cfg.protocol.horizon_schedule = "deadline"
    if args.planner == "cem":
        cfg.planner.n_steps = 10
    spec = OmegaConf.to_container(cfg.environment, resolve=True)
    checkpoint = args.checkpoint.expanduser().resolve()
    inspection = inspect_checkpoint(checkpoint, spec)
    if inspection["model_family"] != "fourier" or inspection["history_frames"] not in (None, 1):
        parser.error("This demo requires a single-image Two-Room Fourier checkpoint")
    config = {
        "demo": True,
        "checkpoint": str(checkpoint),
        "seed": args.seed,
        "steps": args.steps,
        "experiment": OmegaConf.to_container(cfg, resolve=True),
    }
    with RunDirectory(args.output, config) as run:
        model = load_pretrained(checkpoint).to(args.device).eval().requires_grad_(False)
        metadata = checkpoint_metadata(model, checkpoint, spec)
        # Fourier uses checkpoint statistics, so policy construction needs no dataset.
        policy = make_policy(model, metadata, None, cfg)
        env = make_env(spec, args.steps)
        try:
            env.reset(seed=args.seed, options={"variation": []})
            base = env.unwrapped
            start = base.agent_position.cpu().numpy().copy()
            # Coordinates are used only to render the native goal image.
            # The planner receives images and executed actions, never these coordinates.
            base._set_state(base.target_position.cpu().numpy())
            goal = env.render().copy()
            base._set_state(start)
            policy.set_env(
                SimpleNamespace(
                    num_envs=1,
                    single_action_space=env.action_space,
                    action_space=gym.vector.utils.batch_space(env.action_space, 1),
                )
            )
            previous = np.zeros(2, dtype=np.float32)
            records, frames = [], []
            initially_solved = success(env, "tworoom")
            for step in range(args.steps):
                pixels = env.render()
                frames.append(np.concatenate((pixels, goal), axis=1))
                info = {
                    "pixels": pixels[None, None],
                    "goal": goal[None, None],
                    "action": previous[None, None],
                    "terminated": np.array([False]),
                    "id": np.array([[0]]),
                    "step_idx": np.array([[step]]),
                }
                previous = policy.get_action(info)[0]
                if not np.isfinite(previous).all() or not env.action_space.contains(previous):
                    raise ValueError("Planner returned a nonfinite or out-of-bounds action")
                _, _, terminated, truncated, _ = env.step(previous)
                solved = success(env, "tworoom")
                records.append({"step": step + 1, "action": previous.tolist(), "success": solved})
                if solved or terminated or truncated:
                    break
            frames.append(np.concatenate((env.render(), goal), axis=1))
            iio.imwrite(run.path / "rollout.gif", np.stack(frames), duration=100, loop=0)
            atomic_json(run.path / "steps.json", records)
            atomic_json(
                run.path / "results.json",
                {
                    "scope": "One native seeded demo task; not a paper benchmark",
                    "checkpoint": inspection,
                    "success": bool(records[-1]["success"]),
                    "initially_solved": initially_solved,
                    "steps": len(records),
                    "execution_clipping": True,
                    "timing": policy.timings,
                },
            )
            print(f"Saved {run.path / 'rollout.gif'} (current observation left; goal right)")
        finally:
            env.close()


if __name__ == "__main__":
    main()
