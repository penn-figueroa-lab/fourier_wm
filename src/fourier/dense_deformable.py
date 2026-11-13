"""Primitive PyFlex replay. Native targets become causal Cartesian displacements.

The optional simulator lives in a separate Python runtime, so legacy CUDA never
loads alongside Torch. Recording renders directly: FlexEnv.render advances time.
"""

import importlib
import json
import os
import shutil
import subprocess
import sys
import types
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from .runtime import atomic_json, file_hash, fingerprint

DINO_REVISION = "0a9492fa12044b852ae9e001cc74604b79c8bb0c"
SDK_REVISION = "9f002ab35b08727d5de9b220d48b43e52e126eaa"
DT = 1 / 60
FRAMES = 101
BOUND = 0.02


def replay_recipe():
    return {
        "version": 3,
        "source_revision": DINO_REVISION,
        "sdk_revision": SDK_REVISION,
        "recorder_sha256": file_hash(__file__),
        "dt": DT,
        "frames": FRAMES,
        "pushes_per_parent": 20,
        "seed": 42,
        "displacement_bound": BOUND,
        "controls": "Cartesian target minus previous physical-frame pose; native hold=zero",
        "axes": "PyBullet XYZ; scene units; Z is height",
        "sampling": "One seeded contiguous window per native push; reject out-of-bound windows",
        "split": "Original seed-42 parent episodes; never split windows across roles",
        "storage": "RGB and causal commands; no per-frame particle/simulator state",
        "audit_only": [
            "proprio", "command_target", "primitive_index", "source_episode", "source_push"
        ],
    }


def native_runtime():
    root = Path(os.environ["FOURIER_PYFLEX_RUNTIME"]).resolve()
    record = json.loads((root / "runtime.json").read_text())
    if record["dino_revision"] != DINO_REVISION or record["sdk_revision"] != SDK_REVISION:
        raise ValueError("PyFlex source revision differs from the dense replay recipe")
    for name, digest in record["files"].items():
        if file_hash(root / name) != digest:
            raise ValueError(f"PyFlex runtime changed: {name}")
    # Legacy SDK paths use 100-byte buffers; a private short alias avoids overflow.
    import atexit
    import tempfile

    alias = Path(tempfile.mkdtemp(prefix="fwm-flex-")) / "sdk"
    atexit.register(shutil.rmtree, alias.parent, ignore_errors=True)
    alias.symlink_to(root / record["sdk_root"], target_is_directory=True)
    os.environ["PYFLEXROOT"] = str(alias)
    source = root / record["dino_root"]
    package = types.ModuleType("_fourier_dense_native")
    package.__path__ = [str(source / "env/deformable_env/src/sim")]
    sys.modules[package.__name__] = package
    module = importlib.import_module(package.__name__ + ".sim_env.flex_env")
    return module, source


def capture_start(action, speed_inverse, rng, frames=FRAMES):
    h = 1.5  # Both released pusher geometries have pusher_len=1.
    start, end = np.r_[action[:2], h], np.r_[action[2:], h]
    speed = 1 / speed_inverse
    down = int(np.linalg.norm(start - (start + [0, 0, 0.2])) / speed) + 1
    horizontal = int(np.linalg.norm(end - start) / speed) + 1
    return 1 if horizontal < frames else down + 1 + int(rng.integers(horizontal - frames + 1))


class Recorder:
    """Observe PyFlex steps without touching the robot helper's mutable cache."""

    def __init__(self, sim, start, frames=FRAMES):
        import pybullet
        import pyflex

        self.flex, self.bullet, self.sim = pyflex, pybullet, sim
        self.start, self.frames = start, frames
        self.ik_index = self.generation = 0
        self.last_generation = None
        self.target = None
        self.rows, self.actions, self.targets = [], [], []
        self.previous_pose = self.pose()
        self.steps = 0

    def pose(self):
        return np.asarray(
            self.bullet.getLinkState(self.sim.robotId, self.sim.end_idx)[4], np.float64
        )

    def observe(self):
        flex, sim = self.flex, self.sim
        position, angle = flex.get_camPos().copy(), flex.get_camAngle().copy()
        try:
            flex.set_camPos(sim.camPos_list[sim.camera_view])
            flex.set_camAngle(sim.camAngle_list[sim.camera_view])
            raw = flex.render(render_depth=True).reshape(224, 224, 5)
            return raw[..., :3][..., ::-1].astype(np.uint8).copy()
        finally:
            flex.set_camPos(position)
            flex.set_camAngle(angle)

    @contextmanager
    def active(self):
        original_step, original_ik = self.flex.step, self.bullet.calculateInverseKinematics

        def inverse(*args, **kwargs):
            self.ik_index += 1
            self.generation += 1
            self.target = np.asarray(args[2], np.float64).copy()
            return original_ik(*args, **kwargs)

        def step(*args, **kwargs):
            self.steps += 1
            before = self.previous_pose.copy()
            command = (
                np.zeros(3)
                if self.generation == self.last_generation
                else self.target - before
                if self.target is not None
                else np.zeros(3)
            )
            capturing = self.ik_index >= self.start and len(self.rows) < self.frames
            if capturing and self.rows:
                self.actions.append(command.copy())
                self.targets.append(before + command)
            result = original_step(*args, **kwargs)
            self.previous_pose = self.pose()
            if capturing:
                self.rows.append(
                    {
                        "pixels": self.observe(),
                        "proprio": self.previous_pose.copy(),
                        "primitive_index": np.int64(self.steps),
                    }
                )
            self.last_generation = self.generation
            return result

        self.flex.step, self.bullet.calculateInverseKinematics = step, inverse
        try:
            yield self
        finally:
            self.flex.step, self.bullet.calculateInverseKinematics = original_step, original_ik

    def episode(self, parent, push):
        if len(self.rows) != self.frames or len(self.actions) != self.frames - 1:
            return None, "short_window"
        actions = np.asarray(self.actions)
        if not np.isfinite(actions).all() or (np.abs(actions) > BOUND).any():
            return None, "unreachable_or_out_of_bound_command"
        result = {k: np.stack([r[k] for r in self.rows]) for k in self.rows[0]}
        if not all(np.isfinite(v).all() for v in result.values()):
            raise ValueError("Nonfinite primitive replay")
        # The terminal row has no outgoing action. Never replace a used action.
        result["action"] = np.vstack([actions, np.zeros(3)]).astype(np.float32)
        result["command_target"] = np.vstack([self.targets, result["proprio"][-1]])
        result["source_episode"] = np.full(self.frames, parent, np.int64)
        result["source_push"] = np.full(self.frames, push, np.int64)
        return result, None


def append_episode(data, episode):
    """Native SWM HDF5 layout; one complete primitive window per episode."""
    n = len(episode["pixels"])
    offset = len(data["pixels"]) if "pixels" in data else 0
    for name, value in episode.items():
        value = np.asarray(value)
        if name not in data:
            data.create_dataset(
                name,
                shape=(0, *value.shape[1:]),
                maxshape=(None, *value.shape[1:]),
                dtype=value.dtype,
                chunks=(1, *value.shape[1:]),
                compression="lzf",
            )
        column = data[name]
        column.resize(offset + n, axis=0)
        column[offset:] = value
    for name, value in [("ep_len", n), ("ep_offset", offset)]:
        if name not in data:
            data.create_dataset(name, shape=(0,), maxshape=(None,), dtype=np.int64)
        column = data[name]
        column.resize(len(column) + 1, axis=0)
        column[-1] = value
    data.flush()


def truncate(data, windows, frames):
    for key, column in data.items():
        keep = windows if key in {"ep_len", "ep_offset"} else frames
        if len(column) < keep:
            raise ValueError("Replay journal refers to missing committed data")
        column.resize(keep, axis=0)
    data.flush()


def collect(native_root, output, task, *, resume=False, limit=None, parent_budget=None):
    """Optional-runtime process: all physics and rendering occur here."""
    import signal
    import time

    import h5py
    import yaml

    if any(value is not None and value < 1 for value in (limit, parent_budget)):
        raise ValueError("Episode limit and parent budget must be positive")
    module, source = native_runtime()
    spec = yaml.safe_load((source / f"conf/env/{task}.yaml").read_text())
    sim = module.FlexEnv(config=spec)
    recipe = replay_recipe()
    runtime_root = Path(os.environ["FOURIER_PYFLEX_RUNTIME"])
    runtime_hash = file_hash(runtime_root / "runtime.json")
    request = fingerprint(recipe)
    output.mkdir(parents=True, exist_ok=True)
    stopping = [False]
    for sig in [signal.SIGUSR1, signal.SIGTERM]:
        signal.signal(sig, lambda *_: stopping.__setitem__(0, True))
    roles = {}
    processed, started = 0, time.monotonic()
    try:
        for role in ["train", "validation"]:
            original = json.loads((native_root / task / (role + ".json")).read_text())
            selected = original["origin"]["episode_indices"]
            count = min(len(selected), limit) if limit else len(selected)
            dest = output / (role + ".h5")
            journal = output / (role + ".replay.json")
            state = {
                "request": request,
                "parent_cursor": 0,
                "windows": 0,
                "frames": 0,
                "rejected": [],
                "parents": [],
            }
            if journal.exists():
                if not resume:
                    raise FileExistsError("Replay exists; use --resume-data")
                state = json.loads(journal.read_text())
                if state["request"] != request:
                    raise ValueError("Replay recipe changed; use a new dataset directory")
                if state.get("runtime_sha256") != runtime_hash:
                    raise ValueError("Replay runtime changed; preserve the original runtime")
            elif dest.exists():
                raise FileExistsError("Replay data lacks its recovery journal")
            with h5py.File(original["path"], "r") as raw, h5py.File(dest, "a") as data:
                state["runtime_sha256"] = runtime_hash
                truncate(data, state["windows"], state["frames"])
                atomic_json(journal, state)
                for ordinal in range(state["parent_cursor"], count):
                    reserve = 8 if limit else 64
                    if shutil.disk_usage(output).free < reserve * 2**30:
                        raise OSError(
                            f"Dense replay requires {reserve} GiB free reserve; committed parents remain resumable"
                        )
                    parent = selected[ordinal]
                    np.random.seed(42 + parent)
                    rng = np.random.default_rng(np.random.SeedSequence([42, parent]))
                    sim.imgs_list, sim.particle_pos_list, sim.eef_states_list = [], [], []
                    sim.reset(save_data=False)
                    offset = int(raw["ep_offset"][ordinal])
                    positions = raw["state"][offset].reshape(-1, 4)
                    if positions.shape != module.pyflex.get_positions().reshape(-1, 4).shape:
                        raise ValueError("Source particles and runtime topology differ")
                    module.pyflex.set_positions(positions.ravel())
                    module.pyflex.set_velocities(np.zeros((len(positions), 3), np.float32).ravel())
                    actions = raw["action"][offset : offset + recipe["pushes_per_parent"]]
                    windows_before = state["windows"]
                    for push, action in enumerate(actions):
                        start = capture_start(action, sim.robot_speed_inv, rng)
                        recorder = Recorder(sim, start)
                        with recorder.active():
                            result = sim.step(action, save_data=True, data=([], [], []))
                        if result is None:
                            raise ValueError(f"Native replay exploded: {task}/{parent}/{push}")
                        episode, reason = recorder.episode(parent, push)
                        if reason:
                            state["rejected"].append(
                                {"parent": parent, "push": push, "reason": reason}
                            )
                        else:
                            append_episode(data, episode)
                            state["windows"] += 1
                            state["frames"] += FRAMES
                    if state["windows"] == windows_before:
                        raise ValueError(
                            f"No valid bounded window for source parent {parent}; do not silently drop it"
                        )
                    state["parents"].append(parent)
                    state["parent_cursor"] = ordinal + 1
                    data.flush()
                    atomic_json(journal, state)
                    processed += 1
                    remaining = (
                        count
                        - ordinal
                        - 1
                        + (min(100, limit) if limit else 100) * (role == "train")
                    )
                    elapsed = time.monotonic() - started
                    atomic_json(
                        output / "progress.json",
                        {
                            "phase": "dense_replay",
                            "role": role,
                            "parents_done": ordinal + 1,
                            "parents_total": count,
                            "windows": state["windows"],
                            "frames_done": state["frames"],
                            "updated_unix": time.time(),
                            "eta_seconds": elapsed / processed * remaining,
                            "eta_scope": "remaining parents at measured replay rate",
                        },
                    )
                    if stopping[0]:
                        raise SystemExit(75)
                    if parent_budget and processed >= parent_budget and remaining:
                        atomic_json(
                            output / "replay.json",
                            {
                                "status": "partial",
                                "recipe": recipe,
                                "limited": bool(limit),
                                "runtime_sha256": runtime_hash,
                            },
                        )
                        return
            roles[role] = dict(
                **state,
                source_identity=original["identity"],
                sha256=file_hash(dest),
                expected_parent_count=len(selected),
            )
        atomic_json(
            output / "replay.json",
            {
                "status": "completed",
                "recipe": recipe,
                "roles": roles,
                "limited": bool(limit),
                "native_runtime_validated": True,
                "runtime_sha256": runtime_hash,
            },
        )
    finally:
        sim.close()


def prepare_dense(
    native_root, output, environment, *, resume=False, limit=None, parent_budget=None
):
    """Validate native provenance, spawn isolated replay, then finalize manifests."""
    from .data import prepare_manifest, read_manifest

    task = environment["kwargs"]["object_name"]
    native_root, output = Path(native_root).resolve(), Path(output).resolve()
    for role in ["train", "validation"]:
        source = read_manifest(native_root / task / (role + ".json"))
        if source["origin"]["code_revision"] != DINO_REVISION:
            raise ValueError("Dense replay requires the pinned original endpoint release")
    runtime = Path(os.environ["FOURIER_PYFLEX_RUNTIME"]).resolve()
    record = json.loads((runtime / "runtime.json").read_text())
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPATH"] = (
        str(Path(__file__).parents[1]) + os.pathsep + str(runtime / record["module_root"])
    )
    # Keep allocated-node driver paths/preloads. The legacy CUDA runtime must
    # precede them, but replacing them can hide the NVIDIA EGL implementation.
    for key, prefix in {
        "LD_PRELOAD": str(runtime / record["glibc_compat"]),
        "LD_LIBRARY_PATH": os.pathsep.join(
            str(runtime / record[name]) for name in ("cuda_root", "sdl_root")
        ),
    }.items():
        env[key] = prefix + (os.pathsep + env[key] if env.get(key) else "")
    destination = output / environment["name"]
    command = [
        str(runtime / ".venv/bin/python"),
        "-m",
        "fourier.dense_deformable",
        str(native_root),
        str(destination),
        task,
    ]
    if resume:
        command.append("--resume")
    if limit is not None:
        command += ["--limit", str(limit)]
    if parent_budget is not None:
        command += ["--parent-budget", str(parent_budget)]
    import signal

    child = subprocess.Popen(command, env=env)
    previous = {}
    try:
        for sig in [signal.SIGUSR1, signal.SIGTERM]:
            previous[sig] = signal.signal(sig, lambda received, _: child.send_signal(received))
        code = child.wait()
        if code == 75:
            raise SystemExit(75)
        if code:
            raise subprocess.CalledProcessError(code, command)
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    receipt = json.loads((destination / "replay.json").read_text())
    if receipt["recipe"] != replay_recipe():
        raise ValueError("Dense replay receipt has another recipe")
    if receipt["status"] == "partial":
        return receipt
    for role, data in receipt["roles"].items():
        manifest_path = destination / (role + ".json")
        if manifest_path.exists():
            existing = read_manifest(manifest_path)
            if existing["sha256"] != data["sha256"] or existing["origin"]["sha256"] != file_hash(
                destination / "replay.json"
            ):
                raise ValueError("Existing dense manifest differs from completed replay")
            continue
        prepare_manifest(
            destination / (role + ".h5"),
            manifest_path,
            environment,
            source_url="https://github.com/gaoyuezhou/dino_wm#dense-project-replay",
            revision=fingerprint(receipt["recipe"]),
            split=role,
            catalog="custom",
            origin={
                "format": "deformable_dense",
                "path": "replay.json",
                "sha256": file_hash(destination / "replay.json"),
            },
        )
    return receipt


def validate_dense_manifest(manifest, *, allow_limited=False):
    origin = manifest.get("origin", {})
    if origin.get("format") != "deformable_dense":
        raise ValueError("Dense deformable training requires a primitive replay manifest")
    path = Path(manifest["path"]).parent / origin["path"]
    if file_hash(path) != origin["sha256"]:
        raise ValueError("Dense replay receipt changed")
    report = json.loads(path.read_text())
    if (report["limited"] and not allow_limited) or report["status"] != "completed":
        raise ValueError("A limited or incomplete dense replay cannot train a full model")
    if (
        report["recipe"]["source_revision"] != DINO_REVISION
        or report["recipe"]["sdk_revision"] != SDK_REVISION
        or report["recipe"]["dt"] != DT
    ):
        raise ValueError("Dense replay source/control timing differs")
    if not report["limited"] and (
        report["recipe"]["version"] not in {2, 3} or not report.get("runtime_sha256")
    ):
        raise ValueError("Full dense replay requires the audited command/runtime recipe")
    if report['recipe']['version'] == 3:
        required = {'pixels', 'action', 'proprio', 'command_target', 'primitive_index',
                    'source_episode', 'source_push'}
        if set(manifest['columns']) != required:
            raise ValueError('Compact dense replay must contain RGB/commands/audit columns only')
    train, val = report["roles"]["train"], report["roles"]["validation"]
    from .deformable import episode_split

    expected = episode_split(1000)
    for role, data in report["roles"].items():
        parents = data["parents"]
        if (
            not parents
            or parents != expected[role][: len(parents)]
            or (not report["limited"] and len(parents) != len(expected[role]))
        ):
            raise ValueError("Dense replay must preserve the original parent episode split")
    if set(train["parents"]) & set(val["parents"]):
        raise ValueError("Dense replay parent episodes leak across splits")
    if manifest["native_action_statistics"]["outside_declared_space"]:
        raise ValueError("Dense replay commands exceed declared controller bounds")
    role = report["roles"][manifest["split"]]
    if role["sha256"] != manifest["sha256"] or role["windows"] != manifest["episodes"]:
        raise ValueError("Dense replay data/receipt mismatch")
    return report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("native_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("task", choices=["rope", "granular"])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--parent-budget", type=int)
    args = parser.parse_args()
    collect(
        args.native_root,
        args.output,
        args.task,
        resume=args.resume,
        limit=args.limit,
        parent_budget=args.parent_budget,
    )
