"""Import DINO-WM's released push endpoints into SWM without simulator dependencies.

One transition is a whole push, not a fixed-duration physics step. Original
actions, particle states, RGB values and the seeded episode split are retained.
"""

from functools import lru_cache
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import zipfile

import numpy as np

from .download import fetch, file_lock
from .runtime import atomic_json, file_hash, fingerprint


@lru_cache(maxsize=32)
def _digest(path, size, modified, changed, inode):
    return file_hash(path)


def checked_hash(path):
    path = Path(path).resolve()
    stat = path.stat()
    return _digest(str(path), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_ino)


def episode_split(count, fraction=0.9, seed=42):
    import torch

    order = torch.randperm(count, generator=torch.Generator().manual_seed(seed)).tolist()
    n = int(count * fraction)
    return {"train": order[:n], "validation": order[n:]}


def verify_origin(entry, origin, split, episodes, *, base):
    """Check source parts, merged archive and exact source episode selection."""
    if not origin or origin.get("format") != "dinowm_deformable":
        raise ValueError("Deformable imports require their original split-ZIP provenance")
    archive = (Path(base) / origin["path"]).resolve()
    if checked_hash(archive) != origin["sha256"]:
        raise ValueError("Deformable source archive content changed")
    for name, digest in entry["files"].items():
        if checked_hash(archive.parent / name) != digest:
            raise ValueError(f"Deformable release part changed: {name}")
    expected = episode_split(
        entry["released_episodes"], entry["split_fraction"], entry["split_seed"]
    )
    if (
        split not in expected
        or origin["episode_indices"] != expected[split]
        or episodes != len(expected[split])
        or origin["object"] != entry["environment"]
        or origin["code_revision"] != entry["code_revision"]
    ):
        raise ValueError("Deformable episode selection differs from the pinned source split")


def _archive(entry, raw, *, download):
    raw.mkdir(parents=True, exist_ok=True)
    with file_lock(raw / ".archive.lock"):
        for name, digest in entry["files"].items():
            path = raw / name
            if download:
                fetch(
                    f"{entry['source']}/resolve/{entry['revision']}/datasets/{name}",
                    path,
                    digest,
                    progress=raw.parent / "progress.json",
                )
            if not path.is_file() or checked_hash(path) != digest:
                raise ValueError(f"Missing/corrupt pinned part {path}; run prepare-data --download")
        merged = raw / "deformable.full.zip"
        receipt = raw / "combined.json"
        request = fingerprint(entry["files"])
        if receipt.exists():
            saved = json.loads(receipt.read_text())
            if saved["source"] != request or checked_hash(merged) != saved["sha256"]:
                raise ValueError(
                    "Combined ZIP differs from its verified receipt; inspect before replacing it"
                )
            return merged, saved["sha256"]
        if merged.exists():
            raise FileExistsError(
                f"Unverified combined archive {merged}; inspect before replacing it"
            )
        if not shutil.which("zip"):
            raise RuntimeError(
                "Install Info-ZIP 'zip' (e.g. conda install -c conda-forge zip); needed to join split ZIPs"
            )
        if (
            shutil.disk_usage(raw).free
            < sum((raw / n).stat().st_size for n in entry["files"]) + 2**30
        ):
            raise OSError("Joining the release needs another 16 GB free; originals are preserved")
        temporary = raw / ".deformable.joining.zip"
        try:
            temporary.unlink(missing_ok=True)
            subprocess.run(
                ["zip", "-q", "-s", "0", str(raw / "deformable.zip"), "--out", str(temporary)],
                check=True,
                timeout=1800,
                stdin=subprocess.DEVNULL,
            )
            digest = checked_hash(temporary)
            temporary.replace(merged)
            atomic_json(receipt, {"source": request, "sha256": digest})
        finally:
            temporary.unlink(missing_ok=True)
        return merged, digest


def _tensor_file(archive, member, output):
    """Extract only the two declared numeric tensor files, never archive paths."""
    info = archive.getinfo(member)
    receipt = output.with_suffix(".json")
    if output.exists() and receipt.exists():
        saved = json.loads(receipt.read_text())
        if (
            saved["crc32"] == info.CRC
            and saved["size"] == info.file_size
            and checked_hash(output) == saved["sha256"]
        ):
            return output
        raise ValueError(f"Cached source tensor changed: {output}")
    temporary = output.with_suffix(".partial")
    try:
        with archive.open(info) as source, temporary.open("wb") as target:
            shutil.copyfileobj(source, target, length=8 << 20)  # ZipFile checks CRC at EOF.
        digest = checked_hash(temporary)
        temporary.replace(output)
        atomic_json(receipt, {"crc32": info.CRC, "size": info.file_size, "sha256": digest})
    finally:
        temporary.unlink(missing_ok=True)
    return output


def _episode(archive, prefix, index, actions, states, frames, size):
    import torch

    pixels = torch.load(
        io.BytesIO(archive.read(f"{prefix}/{index:06d}/obses.pth")),
        map_location="cpu",
        weights_only=True,
    )
    if not torch.is_tensor(pixels) or tuple(pixels.shape) != (frames, size, size, 3):
        raise ValueError("Released RGB tensor has an unexpected shape")
    # The source saves integer-valued RGB as float32. Store the exact same
    # values as uint8; reject fractional pixels rather than rounding them.
    image = pixels.to(torch.uint8)
    if not torch.equal(pixels, image.to(pixels.dtype)):
        raise ValueError("RGB values must be exact integers in [0,255]; refusing lossy conversion")
    action, state = actions[index].numpy(), states[index].numpy()
    if not np.isfinite(action).all() or not np.isfinite(state).all():
        raise ValueError("Nonfinite released action or particle state")
    return {
        "pixels": image.numpy(),
        "action": action,
        "state": state.reshape(frames, -1),
        "proprio": np.zeros((frames, 1), np.float32),
        "source_episode": np.full(frames, index, np.int32),
    }


def _convert(archive, output, selected, actions, states, entry, request, *, resume):
    """Journal only complete episodes; discard uncommitted tails on resume."""
    from stable_worldmodel.data.formats.hdf5 import HDF5Writer

    partial, journal = output.with_suffix(".partial.h5"), output.with_suffix(".conversion.json")
    committed = 0
    if partial.exists() or journal.exists():
        if not resume:
            raise FileExistsError(f"Conversion exists at {partial}; use --resume-data")
        if not partial.exists() or not journal.exists():
            raise ValueError(
                "Incomplete conversion has no matching journal; inspect it before retrying"
            )
        saved = json.loads(journal.read_text())
        if saved["request"] != request:
            raise ValueError("Conversion request changed; use a different output directory")
        committed = saved["episodes"]
        if not 0 <= committed <= len(selected):
            raise ValueError("Invalid conversion journal")
        if committed:
            _truncate(partial, committed, entry["released_episode_length"])
    initial, started = committed, time.monotonic()
    with HDF5Writer(partial, mode="append" if committed else "overwrite") as writer:
        # Publish a zero-episode journal before the first write too.
        atomic_json(journal, {"request": request, "episodes": committed})
        for ordinal in range(committed, len(selected)):
            episode = _episode(
                archive,
                f"deformable/{entry['environment']}",
                selected[ordinal],
                actions,
                states,
                entry["released_episode_length"],
                entry["image_size"],
            )
            writer.write_episode(episode)
            writer._f.flush()
            elapsed = time.monotonic() - started
            atomic_json(journal, {"request": request, "episodes": ordinal + 1})
            atomic_json(
                output.parent / "progress.json",
                {
                    "phase": "conversion",
                    "split": output.stem,
                    "episodes": ordinal + 1,
                    "total_episodes": len(selected),
                    "eta_seconds": elapsed
                    * (len(selected) - ordinal - 1)
                    / (ordinal + 1 - initial),
                },
            )
    atomic_json(
        journal, {"request": request, "episodes": len(selected), "sha256": checked_hash(partial)}
    )
    partial.replace(output)


def _truncate(partial, committed, frames):
    import h5py

    with h5py.File(partial, "r+") as data:
        if not np.array_equal(
            data["ep_len"][:committed], np.full(committed, frames)
        ) or not np.array_equal(data["ep_offset"][:committed], np.arange(committed) * frames):
            raise ValueError("Committed episode metadata differs from the source")
        for key, column in data.items():
            keep = committed if key in {"ep_len", "ep_offset"} else committed * frames
            if len(column) < keep:
                raise ValueError("Committed conversion data is truncated")
            column.resize(keep, axis=0)


def prepare_release(entry, environment, output, catalog, *, raw=None, resume=False):
    """Download once for both tasks; emit independent SWM train/validation HDF5s."""
    import torch
    from .data import prepare_manifest, read_manifest

    output = Path(output).expanduser().resolve()
    folder = output / environment["name"]
    folder.mkdir(parents=True, exist_ok=True)
    download = raw is None
    raw = Path(raw).expanduser().resolve() if raw else output / "raw"
    with file_lock(folder / ".prepare.lock"):
        archive_path, digest = _archive(entry, raw, download=download)
        splits = episode_split(
            entry["released_episodes"], entry["split_fraction"], entry["split_seed"]
        )
        result = {}
        with zipfile.ZipFile(archive_path) as archive:
            prefix = f"deformable/{environment['name']}"
            actions, states = (
                torch.load(
                    _tensor_file(archive, f"{prefix}/{name}.pth", folder / f"{name}.pth"),
                    weights_only=True,
                    mmap=True,
                    map_location="cpu",
                )
                for name in ("actions", "states")
            )
            count, frames = entry["released_episodes"], entry["released_episode_length"]
            if (
                tuple(actions.shape) != (count, frames, 4)
                or states.shape[:2] != (count, frames)
                or states.shape[-1] != 4
            ):
                raise ValueError("Source action/state shapes differ from the pinned release")
            for role, selected in splits.items():
                path, manifest = folder / f"{role}.h5", folder / f"{role}.json"
                origin = {
                    "format": "dinowm_deformable",
                    "path": os.path.relpath(archive_path, folder),
                    "sha256": digest,
                    "object": environment["name"],
                    "code_revision": entry["code_revision"],
                    "episode_indices": selected,
                    "split_seed": entry["split_seed"],
                    "split_policy": "DINO-WM episode split: torch.randperm seed42, 90% train / 10% validation; no separate test release",
                    "pixels": "float32 integer RGB -> identical uint8 values; no resize or color conversion",
                    "time_unit": "one complete push; primitive durations are not present",
                }
                request = fingerprint(
                    {"entry": entry, "environment": environment, "origin": origin, "role": role}
                )
                if manifest.exists():
                    value = read_manifest(manifest)
                    if (
                        value["environment"] != environment
                        or value.get("origin") != origin
                        or value["catalog"]["id"] != catalog
                    ):
                        raise ValueError(
                            "Existing deformable manifest belongs to a different import"
                        )
                else:
                    if not path.exists():
                        _convert(
                            archive, path, selected, actions, states, entry, request, resume=resume
                        )
                    else:
                        saved = json.loads(path.with_suffix(".conversion.json").read_text())
                        if saved != {
                            "request": request,
                            "episodes": len(selected),
                            "sha256": checked_hash(path),
                        }:
                            raise ValueError(
                                "Unverified converted HDF5; inspect before replacing it"
                            )
                    value = prepare_manifest(
                        path,
                        manifest,
                        environment,
                        source_url=entry["source"],
                        revision=entry["revision"],
                        split=role,
                        catalog=catalog,
                        origin=origin,
                    )
                result[role] = {
                    "manifest": str(manifest),
                    "identity": value["identity"],
                    "episodes": value["episodes"],
                }
        atomic_json(folder / "splits.json", result)
        return result
