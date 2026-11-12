"""Task restoration and observation access, outside the upstream environments.

Privileged state is used only for reset and native success evaluation. It is
never passed to the vision encoder or used as a Fourier training target.
"""

import numpy as np


def row_value(row, key):
    # Lance escapes dots/slashes in some author exports.
    for candidate in (key, key.replace("/", "_"), key.replace(".", "_")):
        if candidate in row:
            return np.asarray(row[candidate])
    raise KeyError(f"Dataset needs reset column {key!r}; refusing an approximate reset")


def button_state(row, index):
    value = row_value(row, f"privileged/button_{index}_state").item()
    if value not in (0, 1):
        raise ValueError("Scene button states must be binary")
    return int(value)  # Author Lance release stores these discrete indices as float32.


def make_env(spec, budget):
    if spec["family"] == "d4rl_maze":
        from .legacy_maze import make_env as legacy_env

        return legacy_env(budget)
    import gymnasium as gym
    import stable_worldmodel  # registers upstream environments

    env = gym.make(
        spec["id"], render_mode="rgb_array", max_episode_steps=budget, **spec.get("kwargs", {})
    )
    if spec.get("rendering", {}).get("hide_goal_marker"):
        if spec["family"] != "fetch":
            env.close()
            raise ValueError("Goal-marker rendering adapter is defined only for Fetch")
        env.unwrapped.model.site("target0").rgba[3] = 0.0
    return env


def validate_action_semantics(env, spec):
    from .actions import ActionSpec

    action = ActionSpec.from_dict(spec["action"])
    base, family = env.unwrapped, spec["family"]
    if env.action_space.shape != (len(action.kinds),):
        raise ValueError("Environment action width differs from configuration")
    if not np.allclose(env.action_space.low, action.low) or not np.allclose(
        env.action_space.high, action.high
    ):
        raise ValueError("Environment native action bounds differ from configuration")
    if family == "pusht":
        if (
            not base.relative
            or not np.allclose(action.scale, base.action_scale)
            or not np.isclose(action.dt, 1 / base.control_hz)
        ):
            raise ValueError("Push-T relative command scale/control interval mismatch")
    elif family == "pointmaze":
        from ogbench.locomaze.point import PointEnv

        # Pinned OGBench PointEnv.step adds 0.2 * action to qpos, then
        # advances five 0.02-second physics steps. Fourier receives the
        # corresponding commanded rate; contact can change achieved motion.
        if not isinstance(base, PointEnv):
            raise ValueError(
                "Fourier PointMaze requires displacement-controlled PointEnv; torque-controlled maze agents are unsupported"
            )
        if (
            action.kinds != ("delta", "delta")
            or not np.allclose(action.scale, 0.2)
            or not np.isclose(action.dt, base.dt)
        ):
            raise ValueError(
                "OGBench PointMaze requires delta commands, scale=0.2 and dt matching the simulator control interval"
            )
    elif family == "d4rl_maze":
        if (
            action.kinds != ("force", "force")
            or not np.allclose(action.scale, 1.0)
            or not np.isclose(action.dt, base.dt)
        ):
            raise ValueError("DINO-WM Maze requires unscaled native motor force and dt=.05")
    elif family in {"cube", "scene"}:
        if not np.allclose(action.scale, base.action_high) or not np.allclose(
            -np.asarray(action.scale), base.action_low
        ):
            raise ValueError("OGBench command channel scales mismatch")
        if family == "scene" and (
            not np.isclose(action.dt, base._control_timestep) or base._mode != "task"
        ):
            raise ValueError("Scene requires the configured control interval and all-object task success")
    elif family in {"tworoom", "simple_pointmaze"}:
        speed = np.asarray(base.variation_space["agent"]["speed"].value).item()
        if not np.allclose(action.scale, speed):
            raise ValueError(
                "Configured command scaling does not match environment speed variation"
            )
    elif family == "fetch":
        if (
            action.kinds != ("delta",) * 4
            or not np.allclose(action.scale, [0.05, 0.05, 0.05, 1.0])
            or not np.isclose(action.dt, base.n_substeps * base.model.opt.timestep)
        ):
            raise ValueError(
                "Fetch command scaling/control interval differs from its native controller"
            )


def restore(env, spec, start, goal, seed):
    """Restore physical state and goal without taking an unrecorded action."""
    family, base = spec["family"], env.unwrapped
    env.reset(seed=int(seed), options={"variation": []})
    if family == "pusht":
        base._set_state(row_value(start, "state"))
        base._set_goal_state(row_value(goal, "state"))
    elif family == "tworoom":
        base._set_state(row_value(start, "proprio")[:2])
        base._set_goal_state(row_value(goal, "proprio")[:2])
    elif family == "simple_pointmaze":
        base.state = row_value(start, "state")[:2].copy()
        base.variation_space["agent"]["position"].set_value(base.state)
        base.variation_space["goal"]["position"].set_value(row_value(goal, "state")[:2])
    elif family == "pointmaze":
        base.set_state(row_value(start, "qpos"), row_value(start, "qvel"))
        base.set_goal(goal_xy=row_value(goal, "qpos")[:2])
    elif family == "d4rl_maze":
        base.set_state(row_value(start, "state"))
        base.set_goal_state(row_value(goal, "state"))
    elif family in {"cube", "scene"}:
        buttons = (
            {
                f"button_state_{i}": button_state(start, i)
                for i in range(base._num_buttons)
            }
            if family == "scene"
            else {}
        )
        base.set_state(row_value(start, "qpos"), row_value(start, "qvel"), **buttons)
        for i in range(base._num_cubes):
            method = base.set_target_pos if family == "cube" else base.set_cube_target_pos
            method(
                i,
                row_value(goal, f"privileged/block_{i}_pos"),
                target_quat=row_value(goal, f"privileged/block_{i}_quat"),
            )
        if family == "scene":
            for i in range(base._num_buttons):
                base._cur_button_states[i] = button_state(start, i)
                base.set_target_button_state(
                    i, button_state(goal, i)
                )
            base._prev_button_states = base._cur_button_states.copy()
            base.set_target_drawer_pos(row_value(goal, "privileged/drawer_pos").item())
            base.set_target_window_pos(row_value(goal, "privileged/window_pos").item())
        base.pre_step()
        base.post_step()
    elif family == "fetch":
        import mujoco

        base.data.qpos[:] = row_value(start, "qpos")
        base.data.qvel[:] = row_value(start, "qvel")
        mujoco.mj_forward(base.model, base.data)
        base.goal = row_value(goal, "achieved_goal").copy()
    else:
        raise ValueError(f"No faithful reset adapter for {family}")
    validate_action_semantics(env, spec)


def state_info(env, family):
    base = env.unwrapped
    if family == "pusht":
        state = np.asarray(base._get_obs())
        return {"state": state, "proprio": np.concatenate((state[:2], state[-2:]))}
    if family == "tworoom":
        return {
            "state": base.agent_position.cpu().numpy().copy(),
            "proprio": base.agent_position.cpu().numpy().copy(),
        }
    if family == "simple_pointmaze":
        return {"state": base.state.copy(), "proprio": base.state.copy()}
    if family == "pointmaze":
        return {
            "state": base.data.qpos.copy(),
            "proprio": base.data.qpos.copy(),
            "qpos": base.data.qpos.copy(),
            "qvel": base.data.qvel.copy(),
        }
    if family == "d4rl_maze":
        return {
            "state": base.state.copy(),
            "proprio": base.state.copy(),
            "qpos": base.state[:2].copy(),
            "qvel": base.state[2:].copy(),
        }
    if family in {"cube", "scene"}:
        info = base.compute_ob_info()
        info["proprio"] = np.concatenate(
            [
                np.atleast_1d(info[k])
                for k in ["proprio/effector_pos", "proprio/effector_yaw", "proprio/gripper_opening"]
            ]
        )
        return info
    if family == "fetch":
        obs = base._get_obs()
        return {
            "proprio": obs["observation"],
            "state": obs["observation"],
            "achieved_goal": obs["achieved_goal"],
            "qpos": base.data.qpos.copy(),
            "qvel": base.data.qvel.copy(),
        }
    raise ValueError(family)


def success(env, family):
    base = env.unwrapped
    if family == "pusht":
        return bool(base.eval_state(base.goal_state, base._get_obs())[0])
    if family == "tworoom":
        return float((base.agent_position - base.target_position).norm()) < 16.0
    if family == "simple_pointmaze":
        return bool(
            np.linalg.norm(base.state - base.variation_space["goal"]["position"].value)
            < base.variation_space["goal"]["radius"].value
        )
    if family == "pointmaze":
        return bool(np.linalg.norm(base.get_xy() - base.cur_goal_xy) <= base._goal_tol)
    if family == "d4rl_maze":
        return bool(np.linalg.norm(base.state[:2] - base.goal_state[:2]) < 0.5)
    if family == "cube":
        values = base._compute_successes()
        return bool(values[base._target_block] if base._mode == "data_collection" else all(values))
    if family == "scene":
        return bool(base._success)
    if family == "fetch":
        return bool(base._is_success(base._get_obs()["achieved_goal"], base.goal))
    raise ValueError(family)


def collect(spec, destination, *, episodes=None, seed=None):
    """Seeded random project collection; published releases are imported separately."""
    from pathlib import Path
    from stable_worldmodel.data.formats.hdf5 import HDF5Writer

    settings = spec["collection"]
    policy = settings["policy"]
    if policy == "official_release":
        raise ValueError(
            "Use the checksum-verified official OGBench navigate release and its visual export; the superseded waypoint collector has been removed."
        )
    if policy != "random":
        raise ValueError(
            "Only explicit random project collection is supported; import published datasets instead"
        )
    seed = settings["seed"] if seed is None else seed
    episodes = settings["episodes"] if episodes is None else episodes
    if Path(destination).exists():
        raise FileExistsError(destination)
    starts = settings.get("start_distribution", "native")
    if starts not in {"native", "uniform_free_cells"} or (
        starts == "uniform_free_cells" and spec["family"] != "pointmaze"
    ):
        raise ValueError("Uniform free-cell collection starts require OGBench PointMaze")
    # Collection termination is recorded separately from evaluation physics.
    # Random world-model data must not lose frames when an incidental goal is hit.
    kwargs = dict(spec.get("kwargs", {}))
    if "terminate_at_goal" in settings:
        if spec["family"] != "pointmaze" or not isinstance(settings["terminate_at_goal"], bool):
            raise ValueError("Collection terminate_at_goal must be a boolean for OGBench PointMaze")
        kwargs["terminate_at_goal"] = settings["terminate_at_goal"]
    env = make_env({**spec, "kwargs": kwargs}, settings["max_episode_steps"])
    try:
        with HDF5Writer(destination, mode="error") as writer:
            for episode in range(episodes):
                # OGBench's task sampler/noise also uses NumPy's global RNG.
                np.random.seed(seed + episode)
                env.action_space.seed(seed + episode)
                rng = np.random.default_rng(seed + episode)
                options = {"variation": []}
                if starts == "uniform_free_cells":
                    cells = np.argwhere(env.unwrapped.maze_map == 0)
                    start, goal = cells[rng.choice(len(cells), 2, replace=False)]
                    options["task_info"] = {"init_ij": start, "goal_ij": goal}
                env.reset(seed=seed + episode, options=options)
                validate_action_semantics(env, spec)
                records = []
                for step in range(settings["max_episode_steps"]):
                    action = env.action_space.sample()
                    record = {
                        **state_info(env, spec["family"]),
                        "pixels": env.render(),
                        "action": action,
                        "seed": np.array(seed + episode),
                        "step_idx": np.array(step),
                        "ep_idx": np.array(episode),
                    }
                    records.append(
                        {
                            k.replace("/", "_"): np.asarray(v).copy()
                            for k, v in record.items()
                            if isinstance(v, (np.ndarray, int, float, np.number))
                        }
                    )
                    _, _, terminated, truncated, _ = env.step(action)
                    if terminated or truncated:
                        break
                writer.write_episode({k: np.stack([r[k] for r in records]) for k in records[0]})
    finally:
        env.close()
