"""Explicit, checksum-pinned dataset downloads and atomic HDF5 extraction."""

from contextlib import contextmanager
import fcntl
from pathlib import Path
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
                    if resumed and not response.headers.get("Content-Range", "").startswith(
                        f"bytes {offset}-"
                    ):
                        raise ValueError("Server returned an incorrect HTTP range")
                    with partial.open("ab" if resumed else "wb") as output:
                        done = offset if resumed else 0
                        initial, started, last = done, time.monotonic(), 0.0
                        length = response.headers.get("Content-Length")
                        total = done + int(length) if length else None
                        while block := response.read(8 << 20):
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
