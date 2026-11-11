"""Saved-task evaluation with native success criteria and explicit protocols."""

import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import torch

from .actions import ActionAdapter, ActionNormalizer
from .data import read_manifest, environment_identity
from .runtime import RunDirectory, atomic_json, fingerprint, file_hash, upstream_module


def action_coordinates(metadata):
    kinds = (metadata.get("action_spec") or {}).get("kinds", [])
    if kinds and all(k == "force" for k in kinds):
        return "native_force"
    if kinds and all(k == "torque" for k in kinds):
        return "native_torque"
    if kinds and any(k in {"force", "torque"} for k in kinds):
        return "mixed_native_force_and_commanded_rate"
    return "commanded_rate" if kinds else "native_or_legacy"


def checkpoint_metadata(model, checkpoint, environment):
    """Read project metadata or adapt an unchanged upstream/legacy checkpoint."""
    path = Path(checkpoint).parent / "training_metadata.json"
    if path.exists():
        return json.loads(path.read_text())
    is_fourier = hasattr(model.predictor, "dynamics") or hasattr(
        model.predictor, "numpy_parameters"
    )
    if is_fourier:
        if not hasattr(model, "action_mean") or not hasattr(model, "action_scale"):
            raise ValueError(
                "This Fourier checkpoint needs its training_metadata.json; evaluation data cannot supply training normalization"
            )
        return {
            "model_family": "fourier",
            "action_block": model.action_block,
            "action_mean": model.action_mean.detach().cpu().tolist(),
            "action_scale": model.action_scale.detach().cpu().tolist(),
            "action_spec": None,
            "metadata_source": "legacy_checkpoint_buffers",
        }
    # Released SWM baselines have config.json and weights, without project metadata.
    config = json.loads((Path(checkpoint).parent / "config.json").read_text())
    family = "dinowm" if getattr(model, "extra_encoders", None) else "lewm"
    width = (
        config["extra_encoders"]["modules"]["action"]["in_chans"]
        if family == "dinowm"
        else config["action_encoder"]["input_dim"]
    )
    primitive_width = len(environment["action"]["kinds"])
    if width % primitive_width:
        raise ValueError("Source checkpoint action width is incompatible with this environment")
    return {
        "model_family": family,
        "action_block": width // primitive_width,
        "action_spec": None,
        "metadata_source": "upstream_config",
        "image_size": environment["image_size"],
    }


def task_list(dataset, identity, protocol, seed, path=None):
    expected = {"dataset_identity": identity, "goal_offset": protocol["goal_offset"], "seed": seed}
    if path:
        record = json.loads(Path(path).read_text())
        if any(record.get(k) != v for k, v in expected.items()):
            raise ValueError("Task list dataset/offset/seed mismatch")
        if len(record["tasks"]) != protocol["num_eval"]:
            raise ValueError("Task list count differs from protocol.num_eval")
        pairs = set()
        for task in record["tasks"]:
            ep, step = task["episode"], task["start"]
            if not (
                0 <= ep < len(dataset.lengths)
                and 0 <= step < dataset.lengths[ep] - protocol["goal_offset"]
            ):
                raise ValueError("Task list contains an invalid episode/start")
            if (ep, step) in pairs:
                raise ValueError("Task list contains duplicate start/goal pairs")
            pairs.add((ep, step))
        return record
    # Match upstream: sample uniformly from valid start rows without replacement.
    rows = [
        (ep, step)
        for ep, n in enumerate(dataset.lengths)
        for step in range(max(0, int(n) - protocol["goal_offset"]))
    ]
    if len(rows) < protocol["num_eval"]:
        raise ValueError("Dataset contains fewer valid start/goal pairs than requested")
    selected = np.sort(
        np.random.default_rng(seed).choice(len(rows), protocol["num_eval"], replace=False)
    )
    return {
        **expected,
        "tasks": [
            {"episode": rows[i][0], "start": rows[i][1], "reset_seed": seed + j}
            for j, i in enumerate(selected)
        ],
    }


def force_maze_tasks(identity, protocol, path=None):
    """Original DINO-WM U-maze state sampler and per-environment seed schedule."""
    seed = protocol["task_seed"]
    tasks = []
    for index in range(protocol["num_eval"]):
        reset_seed = seed * index + 1
        rs = np.random.RandomState(reset_seed)

        def sample():
            while True:
                x, y = rs.uniform(0.5, 3.1), rs.uniform(0.5, 3.1)
                if ((0.5 <= x <= 1.1 or 2.5 <= x <= 3.1) and 0.5 <= y <= 3.1) or (
                    1.1 < x < 2.5 and 2.5 <= y <= 3.1
                ):
                    return [
                        x,
                        y,
                        rs.uniform(-5.2262554, 5.2262554),
                        rs.uniform(-5.2262554, 5.2262554),
                    ]

        tasks.append({"initial_state": sample(), "goal_state": sample(), "reset_seed": reset_seed})
    record = {
        "dataset_identity": identity,
        "task_seed": seed,
        "goal_source": "dinowm_random_state",
        "tasks": tasks,
    }
    if path and json.loads(Path(path).read_text()) != record:
        raise ValueError("Saved force-maze tasks differ from the pinned source sampler")
    return record


def endpoint_row(dataset, episode, step):
    chunk = dataset.load_chunk(np.array([episode]), np.array([step]), np.array([step + 1]))[0]
    row = {}
    for key, value in chunk.items():
        if torch.is_tensor(value):
            value = value.cpu().numpy()
        if isinstance(value, np.ndarray):
            value = value[0]
            if key.startswith("pixels") and value.ndim == 3 and value.shape[0] in (1, 3):
                value = value.transpose(1, 2, 0)
        row[key] = value
    return row


def make_policy(model, metadata, dataset, cfg):
    import stable_worldmodel as swm
    from sklearn.preprocessing import StandardScaler
    from stable_worldmodel.policy import WorldModelPolicy
    from .control import PrimitiveMPCPolicy
    from omegaconf import OmegaConf

    block = int(metadata.get("action_block", 5))
    horizon = cfg.protocol.horizon_steps
    if horizon % block:
        raise ValueError("Physical horizon must be divisible by the checkpoint action block")
    is_fourier = hasattr(model.predictor, "dynamics") or hasattr(
        model.predictor, "numpy_parameters"
    )
    if cfg.protocol.feedback and not is_fourier:
        raise ValueError("TV-LQR feedback is implemented only for Fourier checkpoints")
    paper = cfg.protocol.name in {"swm_table1", "swm_paper", "lewm_paper"}
    image_only = cfg.environment.get("observation_keys") == ["pixels"]
    if image_only and (
        any(k != "action" for k in getattr(model, "extra_encoders", {}))
        or any(k != "pixels" for k in getattr(model, "observation_keys", ("pixels",)))
    ):
        raise ValueError(
            "Image-only benchmark cannot evaluate a checkpoint that consumes proprioception/state"
        )
    if hasattr(model.predictor, "context_state") and (
        cfg.planner.name != "cem" or cfg.protocol.feedback
    ):
        raise ValueError(
            "Image-history dynamics currently require CEM with feedback disabled; single-state SQP/TV-LQR cannot substitute for history-state rollouts"
        )
    history = cfg.history_len or (
        1
        if paper or is_fourier
        else getattr(model, "history_size", getattr(model.predictor, "num_frames", 3))
    )
    if getattr(model.predictor, "history_stride_physical_steps", None) == 1:
        expected = model.predictor.num_frames
        if cfg.history_len is not None and cfg.history_len != expected:
            raise ValueError(
                f"Adjacent history checkpoint requires history_len={expected}; actual frames warm up causally"
            )
        history = expected
    config = swm.PlanConfig(
        horizon=horizon // block,
        action_block=block,
        receding_horizon=max(1, cfg.protocol.replan_steps // block),
        history_len=history,
        warm_start=True,
    )
    process = {}
    adapter = None
    if is_fourier:
        process["action"] = ActionNormalizer(metadata["action_mean"], metadata["action_scale"])
        adapter = ActionAdapter(metadata["action_spec"]) if metadata.get("action_spec") else None
    else:
        # Preserve the pinned upstream evaluator's evaluation-dataset statistics.
        if (image_only and not paper) or getattr(model, "observation_keys", ()) == (
            "pixels",
            "proprio",
        ):
            if "action_mean" not in metadata or "action_scale" not in metadata:
                raise ValueError(
                    "Matched image-only evaluation requires training action normalization metadata"
                )
            process["action"] = ActionNormalizer(metadata["action_mean"], metadata["action_scale"])
        else:
            process["action"] = StandardScaler().fit(
                dataset.get_col_data("action")[np.isfinite(dataset.get_col_data("action")).all(-1)]
            )
    for key in getattr(model, "extra_encoders", {}):
        if key != "action":
            values = dataset.get_col_data(key)
            process[key] = StandardScaler().fit(values[np.isfinite(values).all(-1)])
            process["goal_" + key] = process[key]
    plan_script = upstream_module("plan.eval_wm")
    transform = plan_script.img_transform(
        OmegaConf.create(
            {"eval": {"img_size": metadata.get("image_size", cfg.environment.image_size)}}
        )
    )
    transforms = {"pixels": transform, "goal": transform}
    if getattr(model, "extra_encoders", None):
        objective = swm.planning.WeightedSum(
            [
                (
                    1.0,
                    swm.planning.GoalMSE(
                        "predicted_" + k + "_emb", k + "_goal_emb", reduction="mean"
                    ),
                )
                for k in [*[k for k in model.extra_encoders if k != "action"], "pixels"]
            ]
        )
    else:
        objective = swm.planning.GoalMSE()
    cost = swm.planning.ShootingCostEvaluator(model, objective)
    options = OmegaConf.to_container(cfg.planner, resolve=True)
    name = options.pop("name")
    if name == "cem":
        solver = swm.planning.CEMSolver(
            cost=cost, device=cfg.machine.device, seed=cfg.seed, **options
        )
    else:
        if not is_fourier:
            raise ValueError("The analytic Fourier planner cannot optimize a transformer baseline")
        from .sqp import SWMSQPSolver

        solver = SWMSQPSolver(
            cost=cost, metadata=metadata, device=cfg.machine.device, seed=cfg.seed, **options
        )
    if paper and not is_fourier:
        return WorldModelPolicy(
            solver,
            config,
            process=process,
            transform=transforms,
            history_keys=(
                "pixels",
                *[k for k in getattr(model, "extra_encoders", {}) if k != "action"],
            ),
        )
    return PrimitiveMPCPolicy(
        solver,
        config,
        replan_steps=cfg.protocol.replan_steps,
        process=process,
        transform=transforms,
        adapter=adapter,
        feedback=cfg.protocol.feedback,
        feedback_q=cfg.protocol.get("feedback_q", 1.0),
        feedback_r=cfg.protocol.get("feedback_r", 1.0),
        horizon_schedule=cfg.protocol.get("horizon_schedule", "fixed"),
        clip_actions=not paper,
    )


def reset_callables(dataset, environment):
    """Reuse each source environment's reset recipe; never silently skip state."""
    from omegaconf import OmegaConf
    from .runtime import swm_root

    if environment not in {"pusht", "cube", "tworoom"}:
        raise ValueError(
            "Batched evaluation currently supports pusht/cube/tworoom; use evaluation_batch_size=1 for other environments"
        )
    recipe = OmegaConf.to_container(
        OmegaConf.load(swm_root() / f"scripts/plan/config/{environment}.yaml").eval.callables
    )
    columns = set(dataset.column_names)
    for call in recipe:
        for argument in call.get("args", {}).values():
            if not argument.get("in_dataset", True):
                continue
            name = argument["value"]
            goal = name.startswith("goal_")
            raw = name[5:] if goal else name
            candidates = [raw, raw.replace("privileged_", "privileged/", 1)]
            if environment == "tworoom" and raw == "pos_agent":
                candidates.extend(["proprio", "state"])
            found = next((key for key in candidates if key in columns), None)
            if found is None:
                raise ValueError(f"Faithful {environment} evaluation requires reset column {raw!r}")
            argument["value"] = ("goal_" if goal else "") + found
    return recipe


def evaluate_paper(dataset, tasks, policy, cfg, directory):
    """Use the upstream batched reset/rollout loop without changing its policy."""
    import gymnasium as gym
    import stable_worldmodel as swm
    from omegaconf import OmegaConf
    from .environments import success

    callables = reset_callables(dataset, cfg.environment.name)
    seeds = iter(t["reset_seed"] for t in tasks["tasks"])

    class SeededReset(gym.Wrapper):
        def __init__(self, env):
            super().__init__(env)
            self.fallback_seed = next(seeds)

        def reset(self, *, seed=None, options=None):
            return self.env.reset(
                seed=self.fallback_seed if seed is None else seed, options=options
            )

    world = swm.World(
        cfg.environment.id,
        num_envs=len(tasks["tasks"]),
        image_shape=(224, 224),
        max_episode_steps=2 * cfg.protocol.budget,
        extra_wrappers=[SeededReset],
        **OmegaConf.to_container(cfg.environment.kwargs, resolve=True),
    )
    n = world.num_envs
    steps = np.zeros(n, dtype=int)
    active = np.ones(n, dtype=bool)
    initial, latencies, outside = [], [], np.zeros(n, dtype=int)

    class MeasuredPolicy:
        def set_env(self, env):
            policy.set_env(env)

        def get_action(self, info):
            if not initial:
                initial.extend(success(env, cfg.environment.family) for env in world.envs.envs)
            active[:] &= ~(np.asarray(world.terminateds) | np.asarray(world.truncateds))
            started = time.perf_counter()
            actions = policy.get_action(info)
            if not np.isfinite(actions[active]).all():
                raise ValueError("Nonfinite physical action")
            latencies.append(time.perf_counter() - started)
            steps[:] += active
            outside[:] += active & np.any(
                (actions < world.envs.single_action_space.low)
                | (actions > world.envs.single_action_space.high),
                axis=-1,
            )
            return actions

    world.set_policy(MeasuredPolicy())
    started = time.perf_counter()
    try:
        for call in callables:
            if not hasattr(world.envs.envs[0].unwrapped, call["method"]):
                raise ValueError(f"Native environment lacks required reset method {call['method']}")
        raw = world.evaluate(
            dataset=dataset,
            episodes_idx=[t["episode"] for t in tasks["tasks"]],
            start_steps=[t["start"] for t in tasks["tasks"]],
            goal_offset=cfg.protocol.goal_offset,
            eval_budget=cfg.protocol.budget,
            callables=callables,
            video=directory / "videos" if cfg.output.save_video else None,
        )
        used_seeds = (
            raw["seeds"] if raw["seeds"] is not None else [t["reset_seed"] for t in tasks["tasks"]]
        )
        records = [
            {
                **task,
                "reset_seed": int(used_seeds[i]),
                "success": bool(raw["episode_successes"][i]),
                "initially_solved": initial[i],
                "steps": int(steps[i]),
                "out_of_bounds_actions": int(outside[i]),
            }
            for i, task in enumerate(tasks["tasks"])
        ]
        atomic_json(directory / "episodes.json", records)
        return {
            "success_rate": raw["success_rate"],
            "episodes": n,
            "initially_solved": sum(initial),
            "evaluation_batch_size": n,
            "evaluator": "unchanged_upstream_world",
            "reset_seed_policy": "dataset_seed_or_saved_task_seed",
            "elapsed_seconds": time.perf_counter() - started,
            "batched_policy_seconds": latencies,
            "out_of_bounds_actions": int(outside.sum()),
        }
    except BaseException as error:
        atomic_json(
            directory / "failure.json", {"type": type(error).__name__, "message": str(error)}
        )
        raise
    finally:
        world.close()


def evaluate(cfg):
    import gymnasium as gym
    import stable_worldmodel as swm
    from omegaconf import OmegaConf
    from .environments import make_env, restore, state_info, success

    manifest = read_manifest(cfg.dataset.manifest)
    for key in ("num_eval", "goal_offset", "budget", "horizon_steps", "replan_steps"):
        if not isinstance(cfg.protocol[key], int) or cfg.protocol[key] < 1:
            raise ValueError(f"Set a positive integer protocol.{key} for this benchmark")
    envspec = OmegaConf.to_container(cfg.environment, resolve=True)
    if (
        envspec.get("observation_keys") == ["pixels"]
        and envspec["family"] == "pointmaze"
        and envspec["kwargs"].get("ob_type") != "pixels"
    ):
        raise ValueError(
            "Image-only PointMaze requires native pixel rendering without the task-goal marker"
        )
    if environment_identity(manifest["environment"]) != environment_identity(envspec):
        raise ValueError("Evaluation environment differs from the dataset manifest")
    allowed = cfg.protocol.allowed_environments
    if allowed and cfg.environment.name not in allowed:
        raise ValueError(f"{cfg.protocol.name} supports environments {allowed}")
    paper = cfg.protocol.name in {"swm_table1", "swm_paper", "lewm_paper"}
    if paper and (cfg.planner.name != "cem" or cfg.protocol.feedback):
        raise ValueError("Paper reproduction requires CEM with feedback disabled")
    if paper:
        from .configuration import benchmark_policy

        reference = benchmark_policy(cfg.environment.name)
        if cfg.protocol.name != reference["protocol"] and not (
            cfg.environment.name == "pusht" and cfg.protocol.name == "swm_paper"
        ):
            raise ValueError(
                f"Paper priority selects {reference['protocol']} for this setup; use a project protocol for alternatives"
            )
        from .data import require_paper_dataset

        require_paper_dataset(manifest, cfg.environment.name)
        if manifest["split"] not in {"train", "expert_train"}:
            raise ValueError(
                "This paper preset evaluates the expert training release; label held-out/OOD regimes separately"
            )
        required = {
            "num_eval": 50,
            "goal_offset": 25,
            "budget": 50,
            "horizon_steps": 25,
            "replan_steps": 25,
        }
        if cfg.history_len not in {None, 1}:
            raise ValueError(
                "Pinned source evaluation uses history_len=1; use dataset_goal for a different context"
            )
        iterations = (
            10 if cfg.protocol.name == "lewm_paper" and cfg.environment.name != "pusht" else 30
        )
        if any(cfg.protocol[k] != v for k, v in required.items()) or any(
            cfg.planner[k] != v
            for k, v in {
                "num_samples": 300,
                "topk": 30,
                "n_steps": iterations,
                "var_scale": 1.0,
            }.items()
        ):
            raise ValueError(
                "Modified settings must use dataset_goal or closed_loop, not a paper label"
            )
    batch = cfg.get("evaluation_batch_size") or (cfg.protocol.num_eval if paper else 1)
    if batch not in {1, cfg.protocol.num_eval} or (paper and batch != cfg.protocol.num_eval):
        raise ValueError(
            "evaluation_batch_size must be 1 or num_eval; paper presets require the full batch"
        )
    batched_environment = cfg.environment.name in {"pusht", "cube", "tworoom"}
    if batch > 1 and not batched_environment:
        raise ValueError("This environment requires evaluation_batch_size=1")
    checkpoint = Path(cfg.checkpoint).resolve()
    from .checkpoints import load_pretrained

    model = load_pretrained(checkpoint).to(cfg.machine.device).eval().requires_grad_(False)
    metadata = checkpoint_metadata(model, checkpoint, envspec)
    if paper and metadata["action_block"] != 5:
        raise ValueError("Paper evaluation requires a five-primitive-step action block")
    if metadata.get("action_spec") and fingerprint(metadata["action_spec"]) != fingerprint(
        envspec["action"]
    ):
        raise ValueError("Checkpoint command semantics differ from evaluation environment")
    dataset = swm.data.load_dataset(manifest["path"], num_steps=1, frameskip=1)
    if envspec["family"] == "d4rl_maze":
        tasks = force_maze_tasks(
            manifest["identity"], OmegaConf.to_container(cfg.protocol), cfg.tasks
        )
    else:
        tasks = task_list(
            dataset, manifest["identity"], OmegaConf.to_container(cfg.protocol), cfg.seed, cfg.tasks
        )
    policy = make_policy(model, metadata, dataset, cfg)
    directory = Path(cfg.machine.runs_root).expanduser().resolve() / cfg.run_name
    with RunDirectory(directory, OmegaConf.to_container(cfg, resolve=True)):
        atomic_json(directory / "tasks.json", tasks)
        if batch == cfg.protocol.num_eval and batched_environment:
            result = evaluate_paper(dataset, tasks, policy, cfg, directory)
            result.update(
                checkpoint_sha256=file_hash(checkpoint),
                task_list_sha256=fingerprint(tasks),
                dataset_identity=manifest["identity"],
                protocol=cfg.protocol.name,
                clipped_actions=getattr(policy, "clipped_actions", None),
                execution_clipping=getattr(policy, "clip_actions", False),
                timing=getattr(policy, "timings", {}),
                history_len=policy.cfg.history_len,
                action_coordinates=action_coordinates(metadata),
                normalization="training_checkpoint"
                if metadata["model_family"] == "fourier"
                else "upstream_evaluation_dataset",
            )
            atomic_json(directory / "results.json", result)
            return result
        env = make_env(envspec, cfg.protocol.budget)
        facade = SimpleNamespace(
            num_envs=1,
            single_action_space=env.action_space,
            action_space=gym.vector.utils.batch_space(env.action_space, 1),
        )
        records = []
        policy.set_env(facade)
        try:
            for task_index, task in enumerate(tasks["tasks"]):
                if envspec["family"] == "d4rl_maze":
                    start = {
                        "state": np.asarray(task["initial_state"]),
                        "proprio": np.asarray(task["initial_state"]),
                    }
                    goal = {
                        "state": np.asarray(task["goal_state"]),
                        "proprio": np.asarray(task["goal_state"]),
                    }
                    restore(env, envspec, goal, goal, task["reset_seed"])
                    goal["pixels"] = env.render()
                else:
                    start = endpoint_row(dataset, task["episode"], task["start"])
                    goal = endpoint_row(
                        dataset, task["episode"], task["start"] + cfg.protocol.goal_offset
                    )
                seed = int(np.asarray(start.get("seed", task["reset_seed"])).reshape(-1)[0])
                restore(env, envspec, start, goal, seed)
                if hasattr(policy, "reset"):
                    policy.reset()
                else:
                    policy.set_env(facade)
                initially = success(env, envspec["family"])
                previous = np.zeros(env.action_space.shape, dtype=np.float32)
                frames, latencies = [], []
                solved, steps = initially, 0
                # Keep original budget accounting; initially solved cases are
                # reported separately and still execute like the source evaluator.
                for step in range(cfg.protocol.budget):
                    begin = time.perf_counter()
                    pixels = env.render()
                    if cfg.output.save_video:
                        frames.append(pixels)
                    physical = state_info(env, envspec["family"])
                    info = {
                        "pixels": pixels[None, None],
                        "goal": np.asarray(goal["pixels"])[None, None],
                        "action": previous[None, None],
                        "terminated": np.array([False]),
                        "id": np.array([[task_index]]),
                        "step_idx": np.array([[step]]),
                    }
                    # Explicit allowlist: reset-only state cannot leak into MPC.
                    for key in (
                        () if envspec.get("observation_keys") == ["pixels"] else ("proprio",)
                    ):
                        if key in physical:
                            info[key] = np.asarray(physical[key])[None, None]
                        if key in goal:
                            info["goal_" + key] = np.asarray(goal[key])[None, None]
                    action = policy.get_action(info)[0]
                    if not np.isfinite(action).all():
                        raise ValueError("Nonfinite physical action")
                    latencies.append(time.perf_counter() - begin)
                    _, _, terminated, truncated, returned = env.step(action)
                    previous = action
                    steps += 1
                    solved = success(env, envspec["family"])
                    if (
                        cfg.protocol.get("success_check") == "replan_boundary"
                        and (step + 1) % cfg.protocol.replan_steps
                    ):
                        solved = False
                    if solved or terminated or truncated:
                        break
                record = {
                    **task,
                    "reset_seed": seed,
                    "success": bool(solved),
                    "initially_solved": bool(initially),
                    "steps": steps,
                    "observation_to_action_seconds": latencies,
                }
                records.append(record)
                atomic_json(directory / "episodes.json", records)
                if frames:
                    import imageio.v3 as iio

                    iio.imwrite(
                        directory / f"episode_{len(records):04d}.mp4", np.stack(frames), fps=10
                    )
        except BaseException as error:
            atomic_json(
                directory / "failure.json",
                {
                    "type": type(error).__name__,
                    "message": str(error),
                    "completed_episodes": len(records),
                },
            )
            raise
        finally:
            env.close()
        result = {
            "success_rate": 100 * np.mean([r["success"] for r in records]),
            "episodes": len(records),
            "initially_solved": sum(r["initially_solved"] for r in records),
            "checkpoint_sha256": file_hash(checkpoint),
            "task_list_sha256": fingerprint(tasks),
            "protocol": cfg.protocol.name,
            "clipped_actions": getattr(policy, "clipped_actions", None),
            "evaluation_batch_size": 1,
            "history_len": policy.cfg.history_len,
            "normalization_identity": metadata.get("normalization_identity"),
            "checkpoint_metadata_source": metadata.get(
                "metadata_source", "project_training_metadata"
            ),
            "goal_source": cfg.protocol.get("goal_source", "dataset_offset"),
            "success_check": cfg.protocol.get("success_check", "primitive_step"),
            "budget_status": cfg.protocol.get("budget_status", "declared benchmark budget"),
            "horizon_schedule": getattr(policy, "horizon_schedule", "fixed"),
            "replan_horizons": getattr(policy, "replan_horizons", []),
            "action_coordinates": action_coordinates(metadata),
            "timing": getattr(policy, "timings", {}),
            "normalization": "training_checkpoint"
            if metadata.get("model_family") == "fourier"
            or metadata.get("observation_keys") == ["pixels", "proprio"]
            or envspec.get("observation_keys") == ["pixels"]
            else "upstream_evaluation_dataset",
        }
        atomic_json(directory / "results.json", result)
        return result
