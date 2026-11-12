"""Checksum-pinned releases, atomic extraction and read-only Lance views."""

from contextlib import contextmanager
import fcntl
from pathlib import Path
import re
import shutil
import sys
import tarfile
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .runtime import atomic_json, file_hash


@contextmanager
def file_lock(path):
    with Path(path).open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Another process owns {path}") from error
        yield


def fetch(url, destination, sha256, *, attempts=3, progress=None):
    """Resume bounded HTTP attempts; publish only after checking the full digest."""
    destination = Path(destination)
    if attempts < 1:
        raise ValueError("Download attempts must be positive")
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".partial")
    with file_lock(destination.with_name(destination.name + ".download.lock")):
        if destination.exists():
            if file_hash(destination) != sha256:
                raise ValueError(f"Cached download is corrupt: {destination}. Remove it and rerun.")
            return destination
        for attempt in range(attempts):
            if partial.exists() and file_hash(partial) == sha256:
                partial.replace(destination)
                return destination
            offset = partial.stat().st_size if partial.exists() else 0
            request = Request(url, headers={"Range": f"bytes={offset}-"} if offset else {})
            try:
                with urlopen(request, timeout=60) as response:
                    resumed = response.status == 206
                    length = response.headers.get("Content-Length")
                    total = int(length) if length else None
                    response_end = total
                    if resumed:
                        match = re.fullmatch(
                            r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", "")
                        )
                        if match is None:
                            raise ValueError("Server returned an incorrect HTTP range")
                        start, end, total = map(int, match.groups())
                        if (
                            start != offset
                            or not start <= end < total
                            or (length and int(length) != end - start + 1)
                        ):
                            raise ValueError("Server returned an incorrect HTTP range")
                        response_end = end + 1
                    with partial.open("ab" if resumed else "wb") as output:
                        done = offset if resumed else 0
                        initial, started, last = done, time.monotonic(), 0.0
                        while block := response.read(1 << 20):
                            output.write(block)
                            done += len(block)
                            elapsed = time.monotonic() - started
                            if time.monotonic() - last >= 10 or done == total:
                                speed = (done - initial) / max(elapsed, 1e-6)
                                eta = (total - done) / speed if total and speed else None
                                if progress:
                                    atomic_json(
                                        progress,
                                        {
                                            "phase": "download",
                                            "file": destination.name,
                                            "bytes": done,
                                            "total_bytes": total,
                                            "eta_seconds": eta,
                                        },
                                    )
                                if sys.stderr.isatty():
                                    print(
                                        f"\r{destination.name}: {done / 2**20:.1f} MiB, ETA {eta:.0f}s"
                                        if eta is not None
                                        else f"\r{destination.name}: {done / 2**20:.1f} MiB",
                                        end="",
                                        file=sys.stderr,
                                        flush=True,
                                    )
                                last = time.monotonic()
                    # HTTPResponse.read(size) can return EOF without raising
                    # IncompleteRead. Keep the prefix for a ranged retry;
                    # never mistake a truncated body for corrupt full data.
                    if response_end is not None and done != response_end:
                        raise OSError(
                            f"HTTP body ended at {done}, expected {response_end}; partial retained"
                        )
                    if total is not None and done < total:
                        raise OSError(f"HTTP range ended at {done} of {total}; partial retained")
                if file_hash(partial) != sha256:
                    partial.unlink()
                    raise ValueError(f"Downloaded checksum differs from the pinned release: {url}")
                partial.replace(destination)
                if sys.stderr.isatty():
                    print(file=sys.stderr)
                return destination
            except (OSError, URLError, ValueError) as error:
                if isinstance(error, HTTPError) and error.code == 416:
                    partial.unlink(missing_ok=True)
                if attempt + 1 == attempts:
                    raise RuntimeError(
                        f"Download failed after {attempts} attempts: {url}"
                    ) from error
                time.sleep(attempt + 1)


def unpack_hdf5(archive, destination):
    """Extract only the declared HDF5 member; never materialize archive paths."""
    import zstandard

    archive, destination = Path(archive), Path(destination)
    partial = destination.with_name(destination.name + ".extracting")
    with file_lock(destination.with_name(destination.name + ".extract.lock")):
        if destination.exists():
            return destination  # prepare_manifest verifies correspondence to the archive.
        try:
            with (
                archive.open("rb") as source,
                zstandard.ZstdDecompressor().stream_reader(source) as stream,
            ):
                if archive.name.endswith(".tar.zst"):
                    found = False
                    with tarfile.open(fileobj=stream, mode="r|") as files:
                        for member in files:
                            if member.isfile() and Path(member.name).name == destination.name:
                                if found:
                                    raise ValueError("Archive contains duplicate HDF5 members")
                                with (
                                    files.extractfile(member) as data,
                                    partial.open("wb") as output,
                                ):
                                    shutil.copyfileobj(data, output, length=8 << 20)
                                found = True
                    if not found:
                        raise ValueError(f"Archive does not contain {destination.name}")
                else:
                    with partial.open("wb") as output:
                        shutil.copyfileobj(stream, output, length=8 << 20)
            partial.replace(destination)
        finally:
            partial.unlink(missing_ok=True)
    return destination


def prepare_download(environment, output, *, catalog=None, split=None, resume=False, limit=None):
    """Download declared releases, then delegate their existing provenance checks."""
    output = Path(output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    with file_lock(output / ".prepare.lock"):
        return _prepare_download(
            environment, output, catalog=catalog, split=split, resume=resume, limit=limit
        )


def _prepare_download(environment, output, *, catalog, split, resume, limit):
    from .data import (
        OGBENCH_POINTMAZE_SOURCES,
        dataset_catalog,
        environment_identity,
        import_ogbench_pointmaze,
        prepare_manifest,
        read_manifest,
        require_paper_dataset,
    )

    catalog = catalog or environment["dataset"]
    if environment["name"] == "pointmaze" and catalog in {"custom", "ogbench_pointmaze_navigate"}:
        for name, digest in OGBENCH_POINTMAZE_SOURCES.values():
            fetch(
                "https://rail.eecs.berkeley.edu/datasets/ogbench/" + name,
                output / "raw" / name,
                digest,
                progress=output / "progress.json",
            )
        return import_ogbench_pointmaze(output / "raw", output, resume=resume, limit=limit)
    entries = dataset_catalog()
    entry = entries.get(catalog, {})
    if (
        entry.get("format") == "dinowm_deformable"
        and entry.get("environment") == environment["name"]
    ):
        if split not in {None, "train"} or limit is not None:
            raise ValueError(
                "Deformable releases prepare the complete original train/validation episode split"
            )
        from .deformable import prepare_release

        return prepare_release(entry, environment, output, catalog, resume=resume)
    if entry.get("format") == "lance" and entry.get("environment") == environment["name"]:
        if limit is not None or split not in {None, "train"}:
            raise ValueError("Lance releases prepare complete train/validation/test episode splits")
        return prepare_lance(entry, environment, output, catalog)
    if not entry.get("unpacked") or entry.get("environment") != environment["name"]:
        raise ValueError(
            "No automatic release for this environment/catalog. Use explicit --source and provenance, or --collect for a project benchmark."
        )
    if limit is not None:
        raise ValueError(
            "--episodes limits collection/PointMaze rendering, not a published HDF5 release"
        )
    split = split or ("expert_train" if environment["name"] in {"pusht", "cube"} else "train")
    manifest = output / "train.json"
    if manifest.exists():
        value = read_manifest(manifest)
        require_paper_dataset(value, environment["name"])
        if value["split"] != split or environment_identity(
            value["environment"]
        ) != environment_identity(environment):
            raise ValueError("Existing prepared dataset has a different environment or split")
        return {
            "manifest": str(manifest),
            "identity": value["identity"],
            "episodes": value["episodes"],
        }
    archive = output / "raw" / entry["artifact"]
    url = entry["source"] + "/resolve/" + entry["revision"] + "/" + entry["artifact"]
    fetch(url, archive, entry["archive_sha256"], progress=output / "progress.json")
    source = unpack_hdf5(archive, output / entry["unpacked"])
    value = prepare_manifest(
        source,
        manifest,
        environment,
        source_url=entry["source"],
        revision=entry["revision"],
        split=split,
        catalog=catalog,
        archive=archive,
    )
    return {"manifest": str(manifest), "identity": value["identity"], "episodes": value["episodes"]}


def prepare_lance(entry, environment, output, catalog):
    """Preserve release bytes; shallow clones hold out episodes without image copies."""
    import uuid
    import lance
    import numpy as np
    from .data import prepare_manifest, read_manifest
    from .runtime import tree_hash

    raw = output / "raw" / entry["artifact"]
    for name, digest in entry["files"].items():
        destination = raw / name
        fetch(
            f"{entry['source']}/resolve/{entry['revision']}/{entry['artifact']}/{name}",
            destination,
            digest,
            progress=output / "progress.json",
        )
        # Download locks are not part of the immutable author's dataset.
        destination.with_name(destination.name + ".download.lock").unlink(missing_ok=True)
    if tree_hash(raw) != entry["tree_sha256"]:
        raise ValueError("Lance release tree differs from the pinned file list")
    source = lance.dataset(raw)
    rows = source.to_table(columns=["episode_idx", "step_idx"]).to_pydict()
    ids, steps = np.asarray(rows["episode_idx"]), np.asarray(rows["step_idx"])
    episodes, counts = np.unique(ids, return_counts=True)
    if len(episodes) < 20 or not np.array_equal(ids, np.repeat(episodes, counts)):
        raise ValueError("Lance release must contain at least 20 contiguous complete episodes")
    if not np.array_equal(steps, np.concatenate([np.arange(n) for n in counts])):
        raise ValueError("Lance release has missing or reordered primitive steps")
    order = np.random.default_rng(entry["split_seed"]).permutation(episodes)
    fractions = np.asarray(entry["split_fractions"])
    if len(fractions) != 3 or np.any(fractions <= 0) or not np.isclose(fractions.sum(), 1):
        raise ValueError("Lance split fractions must be three positive fractions summing to one")
    splits = np.split(order, (np.cumsum(fractions)[:2] * len(order)).astype(int))
    result = {}
    for role, selected in zip(("train", "validation", "test"), splits):
        manifest = output / f"{role}.json"
        if manifest.exists():
            value = read_manifest(manifest)
            if (
                value.get("catalog", {}).get("id") != catalog
                or value["environment"] != environment
                or value["source"] != entry["source"]
                or value["revision"] != entry["revision"]
                or value.get("origin", {}).get("sha256") != entry["tree_sha256"]
            ):
                raise ValueError(
                    "Prepared Lance manifest belongs to a different release/environment"
                )
        else:
            view = output / f"{role}.lance"
            if view.exists():
                raise FileExistsError(
                    f"Incomplete view {view}; inspect it before removing and resuming"
                )
            temporary = output / f".prepare-{uuid.uuid4().hex}.lance"
            try:
                clone = source.shallow_clone(temporary, source.version)
                clone.drop_columns(entry["excluded_columns"])
                clone.delete("episode_idx NOT IN (" + ",".join(map(str, selected)) + ")")
                temporary.rename(view)
                value = prepare_manifest(
                    view,
                    manifest,
                    environment,
                    source_url=entry["source"],
                    revision=entry["revision"],
                    split=role,
                    catalog=catalog,
                    origin={
                        "path": str(raw),
                        "sha256": entry["tree_sha256"],
                        "episode_indices": sorted(map(int, selected)),
                        "excluded_columns": entry["excluded_columns"],
                        "split_seed": entry["split_seed"],
                        "version": source.version,
                    },
                )
            finally:
                shutil.rmtree(temporary, ignore_errors=True)
        result[role] = {
            "manifest": str(manifest),
            "identity": value["identity"],
            "episodes": value["episodes"],
        }
    atomic_json(output / "splits.json", result)
    return result
