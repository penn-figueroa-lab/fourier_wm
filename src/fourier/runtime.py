"""Portable source/config discovery and atomic experiment records."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

UPSTREAM_COMMIT = "21446f1ede6d5284e981bd7b47f432b994e6d812"
UPSTREAM_TREE = "b4ea0f6e08ddd22c2fc74344de2f5577b9f563ed"
CONFIG_ROOT = Path(__file__).parent / "configs"


def swm_root():
    import stable_worldmodel

    root = Path(os.environ.get("SWM_ROOT", Path(stable_worldmodel.__file__).resolve().parents[1]))
    if not (root / "scripts/train/lewm.py").exists():
        raise RuntimeError(
            "Install the pinned SWM checkout editable, or set SWM_ROOT to that checkout"
        )
    return root


def upstream_module(name):
    import importlib
    import sys

    root = str(swm_root())
    if root not in sys.path:
        sys.path.insert(0, root)
    return importlib.import_module("scripts." + name)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(value, f, indent=2, allow_nan=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def fingerprint(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def tree_hash(path):
    path = Path(path)
    if path.is_file():
        return file_hash(path)
    return fingerprint(
        [(str(p.relative_to(path)), file_hash(p)) for p in sorted(path.rglob("*")) if p.is_file()]
    )


def verify_upstream():
    """Verify every upstream blob and reconstruct its exact Git tree identity."""
    records = json.loads(Path(__file__).with_name("upstream_files.json").read_text())
    root, changed, tree = swm_root(), [], {}
    for name, record in records.items():
        path = root / name
        if not path.is_file():
            changed.append(name)
            continue
        data = path.read_bytes()
        blob = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
        mode = "100755" if path.stat().st_mode & 0o111 else "100644"
        if blob != record["blob"] or mode != record["mode"]:
            changed.append(name)
        node = tree
        parts = name.split("/")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = (record["mode"], bytes.fromhex(record["blob"]))

    def digest(node):
        entries = []
        for name, value in sorted(
            node.items(), key=lambda kv: kv[0] + ("/" if isinstance(kv[1], dict) else "")
        ):
            mode, sha = ("40000", digest(value)) if isinstance(value, dict) else value
            entries.append(mode.encode() + b" " + name.encode() + b"\0" + sha)
        content = b"".join(entries)
        return hashlib.sha1(b"tree " + str(len(content)).encode() + b"\0" + content).digest()

    manifest_tree = digest(tree).hex()
    return {
        "ok": not changed and manifest_tree == UPSTREAM_TREE,
        "expected_tree": UPSTREAM_TREE,
        "manifest_tree": manifest_tree,
        "changed": changed,
        "checked_files": len(records),
    }


def code_identity():
    package = Path(__file__).resolve().parent
    root = package.parents[1]
    files = {
        str(p.relative_to(package)): file_hash(p)
        for p in sorted(package.rglob("*"))
        if p.is_file() and "__pycache__" not in p.parts
    }
    for name in ("pyproject.toml", "uv.lock"):
        if (root / name).is_file():
            files[name] = file_hash(root / name)
    return fingerprint(files)


def provenance():
    import importlib.metadata
    import platform

    try:
        root = Path(__file__).resolve().parents[2]
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
        diff = subprocess.check_output(["git", "diff", "HEAD"], cwd=root)
    except (subprocess.SubprocessError, OSError):
        revision, diff = "unavailable", b""
        snapshot = Path(__file__).resolve().parents[2] / "SOURCE_REVISION.json"
        if snapshot.exists():
            revision = json.loads(snapshot.read_text())["revision"]
    return {
        "revision": revision,
        "diff_sha256": hashlib.sha256(diff).hexdigest(),
        "code_sha256": code_identity(),
        "upstream_commit": UPSTREAM_COMMIT,
        "python": platform.python_version(),
        "slurm": {
            k: os.environ[k]
            for k in (
                "SLURM_JOB_ID",
                "SLURM_CPUS_PER_TASK",
                "SLURM_MEM_PER_NODE",
                "SLURM_JOB_PARTITION",
                "SLURM_JOB_ACCOUNT",
            )
            if k in os.environ
        },
        "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
    }


class RunDirectory:
    """Exclusive output ownership and immutable requested configuration."""

    def __init__(self, path, config, *, resume=False):
        self.path = Path(path).expanduser().resolve()
        self.path.mkdir(parents=True, exist_ok=resume)
        self.lock = self.path / ".writer.lock"
        self.fd = os.open(self.lock, os.O_CREAT | os.O_RDWR, 0o600)
        import fcntl

        try:
            fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            clean = dict(config)
            clean.pop("resume_from", None)
            record = {"config": clean, "sha256": fingerprint(clean)}
            request = self.path / "request.json"
            if resume:
                previous = json.loads(request.read_text())
                if previous != record:
                    raise ValueError(
                        "Resume configuration differs; use a new run for changed experiments"
                    )
                saved = json.loads((self.path / "provenance.json").read_text())
                if saved.get("code_sha256") != code_identity():
                    raise ValueError("Resume code/dependency identity differs; create a new run")
            else:
                atomic_json(request, record)
                atomic_json(self.path / "provenance.json", provenance())
        except BaseException:
            os.close(self.fd)
            self.fd = None
            raise

    def close(self):
        if getattr(self, "fd", None) is not None:
            os.close(self.fd)
            self.fd = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def __del__(self):
        self.close()
