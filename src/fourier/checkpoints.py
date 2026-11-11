"""Atomic Lightning checkpoints, RNG recovery and graceful Slurm interruption."""

import os
from pathlib import Path
import random
import signal
import tempfile
import time

import lightning as pl
import numpy as np
import torch
from lightning.fabric.plugins.io import TorchCheckpointIO

from .runtime import atomic_json


def load_pretrained(path):
    """Strict SWM loading with the lossless Transformers 4→5 ViT name map."""
    import json
    from hydra.utils import instantiate

    path = Path(path)
    config = json.loads((path.parent / "config.json").read_text())
    model = instantiate(config)
    state = torch.load(path, map_location="cpu", weights_only=True)
    expected = model.state_dict()
    replacements = [
        ("encoder.encoder.layer.", "encoder.layers."),
        (".attention.attention.query.", ".attention.q_proj."),
        (".attention.attention.key.", ".attention.k_proj."),
        (".attention.attention.value.", ".attention.v_proj."),
        (".attention.output.dense.", ".attention.o_proj."),
        (".intermediate.dense.", ".mlp.fc1."),
        (".output.dense.", ".mlp.fc2."),
    ]
    mapped = {}
    for key, value in state.items():
        target = key
        if key not in expected and key.startswith("encoder.encoder.layer."):
            for old, new in replacements:
                target = target.replace(old, new)
        if target in mapped:
            raise ValueError(f"Checkpoint rename collision: {target}")
        mapped[target] = value
    model.load_state_dict(mapped, strict=True)
    return model


def inspect_checkpoint(path, environment):
    """Validate a handoff without a dataset, training, or fitting new statistics."""
    from .evaluation import checkpoint_metadata
    from .runtime import file_hash, fingerprint

    path = Path(path).expanduser().resolve()
    model = load_pretrained(path).eval().requires_grad_(False)
    if any(not torch.isfinite(value).all() for value in model.state_dict().values()):
        raise ValueError("Checkpoint contains non-finite parameters or buffers")
    metadata = checkpoint_metadata(model, path, environment)
    if metadata.get("action_spec") and fingerprint(metadata["action_spec"]) != fingerprint(
        environment["action"]
    ):
        raise ValueError("Checkpoint command semantics differ from the selected environment")
    for key in ("action_mean", "action_scale"):
        if key in metadata:
            value = np.asarray(metadata[key])
            if not np.isfinite(value).all() or (key == "action_scale" and np.any(value <= 0)):
                raise ValueError(f"Checkpoint has invalid {key}")
    return {
        "sha256": file_hash(path),
        "strict_load": True,
        "finite_parameters_and_buffers": True,
        "model_family": metadata["model_family"],
        "seed": metadata.get("seed"),
        "action_block": metadata["action_block"],
        "history_frames": getattr(model.predictor, "num_frames", None),
        "training_normalization_available": "action_mean" in metadata
        and "action_scale" in metadata,
        "scope": "Artifact inspection only; does not measure planning success or convergence",
    }


def training_plugins():
    plugins = [AtomicCheckpointIO()]
    if os.environ.get("SLURM_JOB_ID"):
        from lightning.fabric.plugins.environments import SLURMEnvironment

        # Recovery owns warning signals. Never let Lightning requeue a job.
        plugins.append(SLURMEnvironment(auto_requeue=False))
    return plugins


def resume_path(directory, path):
    if path is not None and not Path(path).resolve().is_relative_to(
        Path(directory).resolve() / "resume"
    ):
        raise ValueError(
            "Exact resume requires this run's own resume checkpoint; use a new run for initialization"
        )
    return path


def metric_logger(directory):
    from lightning.pytorch.loggers import CSVLogger

    # Resume must retain the earlier CSV, not overwrite it.
    root = Path(directory) / "metrics"
    version = len(list(root.glob("segment_*"))) if root.exists() else 0
    return CSVLogger(str(directory), name="metrics", version=f"segment_{version:03d}")


def save_weights(model, directory, config, filename="weights.pt"):
    """Atomic inference artifacts compatible with upstream load_pretrained."""
    from omegaconf import OmegaConf

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    AtomicCheckpointIO().save_checkpoint(model.state_dict(), directory / filename)
    atomic_json(
        directory / "config.json",
        OmegaConf.to_container(config, resolve=True) if OmegaConf.is_config(config) else config,
    )


class InferenceCheckpoint(pl.Callback):
    def __init__(self, directory, config, interval=1):
        self.directory, self.config, self.interval = directory, config, interval

    def on_train_epoch_end(self, trainer, module):
        if trainer.is_global_zero and (trainer.current_epoch + 1) % self.interval == 0:
            save_weights(
                module.model,
                self.directory,
                self.config,
                f"weights_epoch_{trainer.current_epoch + 1}.pt",
            )


class TrainingHealth(pl.Callback):
    """Small progress record for operators; never changes the training objective."""

    def __init__(self, directory):
        self.directory = Path(directory)
        self.started = time.monotonic()
        self.last_write = 0.0

    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
        now = time.monotonic()
        if trainer.global_step > 1 and now - self.last_write < 60:
            return
        loss = outputs.get("loss") if isinstance(outputs, dict) else outputs
        metrics = {
            k: float(v.detach().cpu())
            for k, v in trainer.callback_metrics.items()
            if torch.is_tensor(v) and v.numel() == 1
        }
        if torch.is_tensor(loss) and loss.numel() == 1:
            metrics["batch_loss"] = float(loss.detach().cpu())
        if not all(np.isfinite(v) for v in metrics.values()):
            raise FloatingPointError(
                "Non-finite training metrics; inspect the run's health.json and logs"
            )
        if trainer.is_global_zero:
            atomic_json(
                self.directory / "health.json",
                {
                    "updated_unix": time.time(),
                    "elapsed_seconds": now - self.started,
                    "step": trainer.global_step,
                    "epoch": trainer.current_epoch,
                    "batches_per_epoch": trainer.num_training_batches,
                    "max_epochs": trainer.max_epochs,
                    "metrics": metrics,
                    "learning_rates": [
                        group["lr"] for opt in trainer.optimizers for group in opt.param_groups
                    ],
                    "gpu_peak_bytes": torch.cuda.max_memory_allocated()
                    if torch.cuda.is_available()
                    else None,
                },
            )
        self.last_write = now


class AtomicCheckpointIO(TorchCheckpointIO):
    def save_checkpoint(self, checkpoint, path, storage_options=None):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp = tempfile.mkstemp(prefix=".checkpoint-", suffix=".ckpt", dir=path.parent)
        os.close(fd)
        try:
            super().save_checkpoint(checkpoint, temp, storage_options)
            with open(temp, "rb") as f:
                os.fsync(f.fileno())
            os.replace(temp, path)
        finally:
            if os.path.exists(temp):
                os.unlink(temp)


class Recovery(pl.Callback):
    def __init__(self, directory):
        self.directory = Path(directory)
        self.interrupted = False
        self.previous = {}
        self.rng = None
        self.data_rng = None

    def setup(self, trainer, module, stage):
        for sig in (signal.SIGUSR1, signal.SIGTERM):
            self.previous[sig] = signal.getsignal(sig)
            signal.signal(sig, self._signal)

    def _signal(self, *_):
        self.interrupted = True

    def teardown(self, trainer, module, stage):
        for sig, handler in self.previous.items():
            signal.signal(sig, handler)

    def on_save_checkpoint(self, trainer, module, checkpoint):
        generator = getattr(trainer.train_dataloader, "generator", None)
        checkpoint["fourier_data_rng"] = generator.get_state() if generator is not None else None
        checkpoint["fourier_rng"] = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }

    def on_load_checkpoint(self, trainer, module, checkpoint):
        self.rng = checkpoint.get("fourier_rng")
        self.data_rng = checkpoint.get("fourier_data_rng")
        if self.rng is None:
            raise ValueError(
                "Checkpoint lacks RNG recovery state; use it for initialization, not exact resume"
            )

    def on_train_start(self, trainer, module):
        if self.rng:
            # A resumed Trainer skips sanity validation. Recreate persistent
            # validation workers BEFORE restoring the training RNG, matching
            # the already-initialized workers of uninterrupted execution.
            # The pinned image/column validation transforms are deterministic.
            for loader in trainer.val_dataloaders or []:
                if getattr(loader, "persistent_workers", False) and loader._iterator is None:
                    iter(loader)
        if self.data_rng is not None:
            # Recreating the iterator consumes a worker base seed. Restore the
            # shared sampler/loader generator after its saved iterator is loaded,
            # so the NEXT epoch uses the uninterrupted permutation as well.
            trainer.train_dataloader.generator.set_state(self.data_rng.cpu())
        if self.rng:
            random.setstate(self.rng["python"])
            np.random.set_state(self.rng["numpy"])
            torch.set_rng_state(self.rng["torch"].cpu())
            if self.rng["cuda"] is not None:
                torch.cuda.set_rng_state_all(self.rng["cuda"])

    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
        if self.interrupted:
            trainer.save_checkpoint(self.directory / "resume/last.ckpt")
            trainer.should_stop = True
            if trainer.is_global_zero:
                atomic_json(
                    self.directory / "status.json",
                    {"status": "interrupted", "step": trainer.global_step},
                )
