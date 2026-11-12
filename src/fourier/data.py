"""Dataset provenance and dense command conversion without image duplication."""

import json
from pathlib import Path

import numpy as np

from .actions import ActionAdapter
from .runtime import atomic_json, fingerprint, tree_hash, file_hash, load_catalog


def dataset_catalog():
    from omegaconf import OmegaConf

    return OmegaConf.to_container(load_catalog().datasets)


def require_paper_dataset(manifest, environment):
    catalog = dataset_catalog()
    expected = next((key for key, v in catalog.items() if v["environment"] == environment), None)
    proof = manifest.get("catalog", {})
    if expected is None or proof.get("id") != expected or not proof.get("verified"):
        raise ValueError(
            f"Paper protocol requires the verified {expected or environment} release. Prepare it with --catalog; use training_profile=project for custom data."
        )
    entry = catalog[expected]
    if manifest["source"] != entry["source"] or manifest["revision"] != entry["revision"]:
        raise ValueError("Paper dataset source/revision differs from the pinned catalog")
    return entry


def archive_content_hash(path, expected_name):
    """Link a verified compressed source to the exact extracted HDF5 bytes."""
    import hashlib
    import tarfile
    import zstandard

    digest = hashlib.sha256()
    with open(path, "rb") as source, zstandard.ZstdDecompressor().stream_reader(source) as stream:
        if ".tar." in str(path):
            with tarfile.open(fileobj=stream, mode="r|") as archive:
                matches = 0
                for entry in archive:
                    if entry.isfile() and Path(entry.name).name == expected_name:
                        matches += 1
                        extracted = archive.extractfile(entry)
                        for chunk in iter(lambda: extracted.read(8 << 20), b""):
                            digest.update(chunk)
                        break
                if matches != 1:
                    raise ValueError(f"Archive must contain {expected_name}")
        else:
            for chunk in iter(lambda: stream.read(8 << 20), b""):
                digest.update(chunk)
    return digest.hexdigest()


def environment_identity(environment):
    identity = {k: environment[k] for k in ("id", "family", "kwargs", "image_size", "action")}
    if "rendering" in environment:
        identity["rendering"] = environment["rendering"]
    return identity


def prepare_manifest(
    source,
    output,
    environment,
    *,
    source_url,
    revision,
    split,
    expected_sha256=None,
    catalog=None,
    archive=None,
    origin=None,
):
    import stable_worldmodel as swm

    path = Path(source).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset missing: {path}. Download/collect the declared source first."
        )
    if not source_url or not revision:
        raise ValueError("Specify dataset provenance: source URL/collection recipe and revision")
    ds = swm.data.load_dataset(str(path), frameskip=1, num_steps=1)
    adapter = ActionAdapter(environment["action"])
    raw = ds.get_col_data("action")
    q = ds.get_col_data(adapter.spec.position_key) if adapter.state_dependent else None
    converted = adapter.to_model(raw, q)
    terminals = np.asarray(ds.offsets) + np.asarray(ds.lengths) - 1
    valid = np.isfinite(converted).all(-1)
    invalid = np.flatnonzero(~valid)
    if len(np.setdiff1d(invalid, terminals)):
        raise ValueError("Nonfinite action/proprioception outside terminal records")
    outside = (
        (raw[valid] < np.asarray(adapter.spec.low) - 1e-5)
        | (raw[valid] > np.asarray(adapter.spec.high) + 1e-5)
    ).any(-1)
    checksum = tree_hash(path)
    if expected_sha256 and checksum != expected_sha256:
        raise ValueError("Dataset checksum does not match the expected source artifact")
    proof = {"id": "custom", "verified": False}
    collection = {"kind": "project_collection_defaults", **environment.get("collection", {})}
    if catalog and catalog != "custom":
        entry = dataset_catalog()[catalog]
        if (
            entry["environment"] != environment["name"]
            or source_url != entry["source"]
            or revision != entry["revision"]
        ):
            raise ValueError(
                "Dataset environment/source/revision differs from the pinned paper release"
            )
        approved = entry.get("approved_unpacked_by_split", {}).get(
            split, entry.get("approved_unpacked_sha256", [])
        )
        verified = checksum in approved
        if entry.get("format") == "lance":
            verified = bool(
                origin and origin["sha256"] == entry["tree_sha256"]
                and tree_hash(origin["path"]) == origin["sha256"]
                and origin["excluded_columns"] == entry["excluded_columns"]
            )
            if not verified or len(ds.lengths) != len(origin["episode_indices"]):
                raise ValueError("Lance view requires its verified immutable source and episode selection")
        if not verified and archive:
            if file_hash(archive) != entry["archive_sha256"]:
                raise ValueError("Compressed source differs from the author release checksum")
            verified = archive_content_hash(archive, path.name) == checksum
        if not verified:
            raise ValueError(
                "Unverified paper data: provide --archive for the pinned author archive; official OGBench NPZ/custom collections cannot substitute for LeWM data"
            )
        expected_episodes = entry.get("split_episodes", {}).get(split, entry.get("released_episodes"))
        if (expected_episodes is not None and len(ds.lengths) != expected_episodes) or (
            "released_frames" in entry and sum(ds.lengths) != entry["released_frames"]
        ):
            raise ValueError("Dataset episode/frame counts differ from the pinned release")
        # Transition counts in papers can exclude the terminal observation.
        # Validate stored rows against the inspected release specification.
        if "released_episode_length" in entry and not np.all(
            np.asarray(ds.lengths) == entry["released_episode_length"]
        ):
            raise ValueError("Dataset episode lengths differ from the inspected release")
        proof = {"id": catalog, "verified": True}
        if "archive_sha256" in entry:
            proof["archive_sha256"] = entry["archive_sha256"]
        elif origin:
            proof["source_tree_sha256"] = origin["sha256"]
        collection = {
            "kind": "author_release",
            "policy": entry["collection_policy"],
            "episodes": len(ds.lengths),
            "seed": None,
            "seed_status": "not specified by release metadata",
        }
    from importlib.metadata import version
    from .runtime import UPSTREAM_COMMIT

    result = {
        "schema_version": 2,
        "path": str(path),
        "sha256": checksum,
        "source": source_url,
        "revision": revision,
        "split": split,
        "environment": environment,
        "episodes": len(ds.lengths),
        "lengths": np.asarray(ds.lengths).tolist(),
        "offsets": np.asarray(ds.offsets).tolist(),
        "columns": list(ds.column_names),
        "action_dim": raw.shape[-1],
        "terminal_nonfinite_rows": invalid.tolist(),
        "native_action_statistics": {
            "minimum": raw[valid].min(0).tolist(),
            "maximum": raw[valid].max(0).tolist(),
            "outside_declared_space": int(outside.sum()),
            "training_actions_clipped": False,
        },
        "conversion": "lazy_dense_before_subsampling",
        "native_data_modified": False,
        "catalog": proof,
        "collection_settings": collection,
        "environment_collection_role": "template for explicit project collection, not an imported artifact's recipe",
        "environment_version": {
            "swm_commit": UPSTREAM_COMMIT,
            **{name: version(name) for name in ("gymnasium", "mujoco", "ogbench", "pymunk")},
        },
        "image_preprocessing": {
            "implementation": "pinned upstream model preprocessor",
            "target_size": environment["image_size"],
            "source_images_preserved": True,
        },
        "split_policy": "artifact split; baseline training uses the source random clip split unless external validation is supplied",
    }
    if origin:
        result["origin"] = origin
        result["split_policy"] = "fixed disjoint episode holdout: 90% train, 5% validation, 5% test; no image/action rewrite"
    result["identity"] = fingerprint({k: v for k, v in result.items() if k != "path"})
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    atomic_json(output, result)
    return result


def read_manifest(path, *, verify=True):
    value = json.loads(Path(path).read_text())
    identity = value.pop("identity")
    identity_fields = (
        {k: v for k, v in value.items() if k != "path"}
        if value.get("schema_version", 1) >= 2
        else value
    )
    if fingerprint(identity_fields) != identity:
        raise ValueError("Dataset manifest was modified")
    value["identity"] = identity
    if verify and tree_hash(value["path"]) != value["sha256"]:
        raise ValueError("Dataset content differs from the verified manifest")
    if verify and value.get("origin"):
        origin = value["origin"]
        if tree_hash(origin["path"]) != origin["sha256"]:
            raise ValueError("Lance view source content differs from its pinned release")
    return value


def command_dataset(name, action_spec, **kwargs):
    """Delegate storage to SWM while converting primitive commands causally."""
    return DenseCommandDataset(name, action_spec, **kwargs)


class DenseCommandDataset:
    """Storage-reader proxy with SWM's original clip indexing and transforms."""

    def __init__(self, name, action_spec, **kwargs):
        from stable_worldmodel.data.dataset import Dataset
        from .loading import load_dataset

        self.adapter = ActionAdapter(action_spec)
        self.transform = kwargs.pop("transform", None)
        self.frameskip = int(kwargs.pop("frameskip", 1))
        self.num_steps = int(kwargs.pop("num_steps", 1))
        self.history_frames = int(kwargs.pop("history_frames", 1))
        if self.history_frames < 1 or (self.history_frames > 1 and self.num_steps < 2):
            raise ValueError(
                "Adjacent history requires positive history_frames and at least two endpoints"
            )
        self.span = self.frameskip * self.num_steps
        keys = kwargs.get("keys_to_load")
        if keys is not None and self.adapter.state_dependent:
            kwargs["keys_to_load"] = list(dict.fromkeys([*keys, self.adapter.spec.position_key]))
        self.raw = load_dataset(name, frameskip=1, num_steps=1, **kwargs)
        self.lengths, self.offsets = self.raw.lengths, self.raw.offsets
        index = Dataset(self.lengths, self.offsets, self.frameskip, self.num_steps)
        self.clip_indices = index.clip_indices

    @property
    def column_names(self):
        return self.raw.column_names

    def __len__(self):
        return len(self.clip_indices)

    def __getitem__(self, idx):
        import torch
        from .loading import SelectedFrameDataset

        ep, start = self.clip_indices[idx]
        begin = max(0, start - self.history_frames + 1)
        endpoints = torch.arange(self.num_steps) * self.frameskip
        offsets = endpoints
        if self.history_frames > 1:
            # Encode only frames needed by teacher-forced source histories and
            # sparse targets. Clip indexing and endpoint offsets stay upstream's.
            sources = endpoints[:-1, None] - torch.arange(self.history_frames)
            offsets = torch.unique(torch.cat((sources.flatten(), endpoints)), sorted=True)
            history_indices = torch.searchsorted(offsets, sources)
            endpoint_indices = torch.searchsorted(offsets, endpoints)
        pixel_indices = (start + offsets).clamp_min(0)
        if isinstance(self.raw, SelectedFrameDataset):
            values = self.raw._load_slice(
                ep, begin, start + self.span, pixel_indices=pixel_indices.tolist()
            )
            pixels = values.pop("pixels")
        else:
            values = self.raw._load_slice(ep, begin, start + self.span)
            pixels = values.pop("pixels")[pixel_indices - begin]
        q = values.get(self.adapter.spec.position_key)
        values["action"] = self.adapter.to_model(values["action"], q)
        values = {k: v if isinstance(v, str) else v[start - begin :] for k, v in values.items()}
        if not torch.isfinite(values["action"][: self.span - self.frameskip]).all():
            raise ValueError("Nonfinite supervised primitive commands")
        values = {
            k: v if k == "action" or isinstance(v, str) else v[:: self.frameskip]
            for k, v in values.items()
        }
        values["pixels"] = pixels
        if self.history_frames > 1:
            values.update(history_indices=history_indices, endpoint_indices=endpoint_indices)
        if self.transform:
            values = self.transform(values)
        values["action"] = values["action"].reshape(self.num_steps, -1)
        return values

    def get_col_data(self, col):
        values = self.raw.get_col_data(col)
        if col == "action":
            q = (
                self.raw.get_col_data(self.adapter.spec.position_key)
                if self.adapter.state_dependent
                else None
            )
            values = self.adapter.to_model(values, q)
        return values

    def get_dim(self, col):
        return self.raw.get_dim(col)


OGBENCH_POINTMAZE_SOURCES = {
    "train": (
        "pointmaze-large-navigate-v0.npz",
        "82a73ed8de90ad2b8bf89069253e961918438d127d7f7c7f09c81e642d0d2c61",
    ),
    "validation": (
        "pointmaze-large-navigate-v0-val.npz",
        "19e6b510c800b865d5c3f9e4335941a602731184fa683eb99face7c5405b6ec4",
    ),
}


def import_ogbench_pointmaze(
    source_root, output_root, *, resume=False, limit=None, stop_after=None
):
    """Export the pinned OGBench navigate release through SWM's native HDF5 format.

    Takes a directory containing the two original NPZ files. Output is separate
    from raw data. SIGUSR1/SIGTERM flush a resumable frame cursor and exit 75.
    A limited export is a smoke fixture and cannot satisfy the default training
    protocol. This renderer preserves recorded states/actions; it never steps
    the simulator to manufacture observations.
    """
    import fcntl
    import signal

    source_root, output_root = (
        Path(source_root).expanduser().resolve(),
        Path(output_root).expanduser().resolve(),
    )
    if limit is not None and limit < 1:
        raise ValueError("Episode limit must be positive")
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        previous = {sig: signal.getsignal(sig) for sig in (signal.SIGUSR1, signal.SIGTERM)}
        try:
            return _import_ogbench_pointmaze(
                source_root, output_root, resume=resume, limit=limit, stop_after=stop_after
            )
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)


def _import_ogbench_pointmaze(
    source_root, output_root, *, resume=False, limit=None, stop_after=None
):
    import h5py
    import mujoco
    import signal
    import time
    from omegaconf import OmegaConf
    from .configuration import compose
    from .environments import make_env
    from .runtime import code_identity

    output_root.mkdir(parents=True, exist_ok=True)
    spec = OmegaConf.to_container(compose(overrides=["environment=pointmaze"]).environment)
    spec["collection"] = {
        "policy": "official_ogbench_navigate",
        "max_episode_steps": 1001,
        "seed": None,
        "seed_status": "not specified by released artifact",
        "direction": "unit waypoint direction",
        "action_noise_std": 0.5,
        "terminate_at_goal": False,
        "goal_resampling": "after success",
    }
    request = {
        "export_code_sha256": code_identity(),
        "sources": OGBENCH_POINTMAZE_SOURCES,
        "environment": spec,
        "episode_limit": limit,
        "source_repository_revision": "6eda27402c69681cfe584cc9f3a3fe2a24ec9163",
        "source_urls": {
            role: "https://rail.eecs.berkeley.edu/datasets/ogbench/" + v[0]
            for role, v in OGBENCH_POINTMAZE_SOURCES.items()
        },
    }
    identity = fingerprint(request)
    request_path = output_root / "request.json"
    if request_path.exists():
        if not resume:
            raise FileExistsError("Existing export requires explicit --resume")
        if json.loads(request_path.read_text()) != json.loads(json.dumps(request)):
            raise ValueError("Resume source/config differs")
    else:
        atomic_json(request_path, request)
    stopping = {"value": False}

    def stop(*_):
        stopping["value"] = True

    for sig in [signal.SIGUSR1, signal.SIGTERM]:
        signal.signal(sig, stop)
    manifests = {}
    total_done = 0
    for role, (name, checksum) in OGBENCH_POINTMAZE_SOURCES.items():
        archive = source_root / name
        if file_hash(archive) != checksum:
            raise ValueError("Official source checksum mismatch: " + name)
        source = np.load(archive)
        ends = np.flatnonzero(source["terminals"])
        starts = np.r_[0, ends[:-1] + 1]
        lengths = ends - starts + 1
        expected_episodes = 1000 if role == "train" else 100
        if len(ends) != expected_episodes or not np.all(lengths == 1001):
            raise ValueError("Official release episode boundaries differ")
        count = len(ends) if limit is None else min(limit, len(ends))
        lengths = lengths[:count]
        n = int(lengths.sum())
        ends = ends[:count]
        path = output_root / (role + ".h5")
        partial = path.with_suffix(".h5.partial")
        # NpzFile does not cache members: indexing it inside the render loop
        # decompresses the full million-row column for every frame.
        qpos, qvel = source["qpos"][:n], source["qvel"][:n]
        actions = source["actions"][:n].copy()
        # The final stored state has no recorded next observation. Its original
        # action remains in the immutable NPZ; exclude it from model supervision.
        actions[ends] = np.nan
        if not path.exists():
            env = make_env(spec, 1001)
            try:
                if not partial.exists():
                    with h5py.File(partial, "w") as f:
                        f.attrs.update(request_identity=identity, rendered_frames=0)
                        f.create_dataset("ep_len", data=lengths)
                        f.create_dataset("ep_offset", data=np.r_[0, np.cumsum(lengths)[:-1]])
                        for key, values in {
                            "qpos": qpos,
                            "qvel": qvel,
                            "action": actions,
                            "ep_idx": np.repeat(np.arange(count), lengths),
                            "step_idx": np.tile(np.arange(1001), count),
                        }.items():
                            f.create_dataset(key, data=values)
                        f.create_dataset(
                            "pixels",
                            shape=(n, 224, 224, 3),
                            dtype=np.uint8,
                            chunks=(1, 224, 224, 3),
                            compression="lzf",
                        )
                with h5py.File(partial, "r+") as f:
                    if f.attrs["request_identity"] != identity:
                        raise ValueError("Partial export belongs to another request")
                    cursor = int(f.attrs["rendered_frames"])
                    begin = time.monotonic()
                    last = 0.0
                    env.reset(seed=42, options={"variation": []})
                    base = env.unwrapped
                    for index in range(cursor, n):
                        base.set_state(qpos[index], qvel[index])
                        mujoco.mj_forward(base.model, base.data)
                        f["pixels"][index] = env.render()
                        if time.monotonic() - last >= 10 or index + 1 == n or stopping["value"]:
                            f.attrs["rendered_frames"] = index + 1
                            f.flush()
                            elapsed = time.monotonic() - begin
                            completed = index + 1 - cursor
                            atomic_json(
                                output_root / "progress.json",
                                {
                                    "phase": "render_official",
                                    "role": role,
                                    "frames_done": index + 1,
                                    "frames_total": n,
                                    "updated_unix": time.time(),
                                    "eta_seconds": (n - index - 1) * elapsed / completed
                                    if completed
                                    else None,
                                },
                            )
                            print(f"OFFICIAL_RENDER {role} {index + 1}/{n}", flush=True)
                            last = time.monotonic()
                        total_done += 1
                        if stopping["value"] or (
                            stop_after is not None and total_done >= stop_after
                        ):
                            f.attrs["rendered_frames"] = index + 1
                            f.flush()
                            raise SystemExit(75)
            finally:
                renderer = getattr(env.unwrapped, "custom_renderer", None)
                if renderer is not None:
                    renderer.close()
                env.close()
            partial.replace(path)
        with h5py.File(path, "r") as f:
            for key in ["qpos", "qvel"]:
                if not np.array_equal(f[key][:], source[key][:n]):
                    raise ValueError("Physical states changed")
            if not np.array_equal(f["action"][:], actions, equal_nan=True):
                raise ValueError("Command alignment changed")
        output_spec = json.loads(json.dumps(spec))
        output_spec["collection"] = {
            "policy": "official_ogbench_navigate",
            "episodes": count,
            "max_episode_steps": 1001,
            "seed": None,
            "seed_status": "not specified by released artifact",
            "direction": "unit waypoint direction",
            "action_noise_std": 0.5,
            "terminate_at_goal": False,
            "goal_resampling": "after success",
            "record_count_per_episode": 1001,
            "usable_transitions_per_episode": 1000,
        }
        manifest_path = output_root / (role + ".json")
        if not manifest_path.exists():
            prepare_manifest(
                path,
                manifest_path,
                output_spec,
                source_url=request["source_urls"][role],
                revision="sha256:" + checksum,
                split=role,
                catalog="custom",
            )
        manifests[role] = read_manifest(manifest_path)
        atomic_json(
            output_root / "conversion.json",
            {
                "request_identity": identity,
                "code_sha256": code_identity(),
                "raw_source_sha256": {k: v[1] for k, v in OGBENCH_POINTMAZE_SOURCES.items()},
                "outputs": {k: v["identity"] for k, v in manifests.items()},
                "state_columns_bitwise_preserved": True,
                "nonterminal_actions_bitwise_preserved": True,
                "terminal_action": "NaN in export; original NPZ preserved unchanged",
                "rendering": "native pixels, no task-goal marker; no stepping or future state used",
            },
        )
    atomic_json(
        output_root / "ready.json",
        {
            "roles": {
                k: {"identity": v["identity"], "episodes": v["episodes"], "sha256": v["sha256"]}
                for k, v in manifests.items()
            },
            "ready_unix": time.time(),
            "source": "official OGBench pointmaze-large-navigate-v0",
            "limited_pilot": limit is not None,
        },
    )

    return json.loads((output_root / "ready.json").read_text())
