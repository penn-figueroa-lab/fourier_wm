"""Endpoint prediction + optional SIGReg, independent of encoder/data format."""

import torch
from torch import nn


class SIGReg(nn.Module):
    """LeWM's single-device sketch Gaussian regularizer on [T, B, D].

    Adapted from LeWM module.py (Lucas Maes, MIT); see stable-worldmodel/LICENSE.
    This is not a guarantee against all representation collapse.
    """

    def __init__(self, knots=17, num_proj=1024):
        super().__init__()
        if knots < 2 or num_proj < 1:
            raise ValueError("Require knots>=2 and num_proj>=1")
        self.num_proj = num_proj
        t = torch.linspace(0, 3, knots)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt)
        weights[[0, -1]] = dt
        self.register_buffer("t", t)
        self.register_buffer("phi", torch.exp(-t.square() / 2))
        self.register_buffer("weights", weights * self.phi)

    def forward(self, embeddings):
        if embeddings.ndim != 3 or embeddings.shape[1] < 2:
            raise ValueError("Expected [T, B>=2, D]")
        directions = torch.randn(embeddings.shape[-1], self.num_proj, device=embeddings.device)
        directions = directions / directions.norm(dim=0).clamp_min(1e-12)
        phase = (embeddings @ directions).unsqueeze(-1) * self.t
        error = (phase.cos().mean(-3) - self.phi).square() + phase.sin().mean(-3).square()
        return ((error @ self.weights) * embeddings.shape[-2]).mean()


def endpoint_loss(dynamics, embeddings, controls):
    """[B,T,D], [B,T-1,H,m]; mean endpoint MSE across B, T-1, and D.

    H primitive transitions connect adjacent encoded observations. Intermediate
    predicted states are NOT supervised. Neither sources nor targets detach.
    """
    if (
        embeddings.ndim != 3
        or embeddings.shape[1] < 2
        or controls.ndim != 4
        or controls.shape[:2] != (embeddings.shape[0], embeddings.shape[1] - 1)
    ):
        raise ValueError("Expected [B,T>=2,D] and [B,T-1,H,m]")
    predicted = dynamics.rollout(embeddings[:, :-1], controls)[..., -1, :]
    return (predicted - embeddings[:, 1:]).square().mean()


def world_model_loss(dynamics, embeddings, controls, regularizer, sigreg_weight=0.09):
    """Train any encoder/projector jointly with Fourier dynamics."""
    if sigreg_weight < 0:
        raise ValueError("SIGReg weight must be nonnegative")
    forward = endpoint_loss(dynamics, embeddings, controls)
    regularization = regularizer(embeddings.transpose(0, 1))
    loss = forward + sigreg_weight * regularization
    return loss, {"forward": forward.detach(), "sigreg": regularization.detach()}


# Project-owned integration glue. The definitions above are the supplied code.
def swm_forward(self, batch, stage, cfg):
    from .world_model import LEWM_TARGETS

    if cfg.model._target_ == "fourier.world_model.AdjacentHistoryWorldModel":
        output = self.model.encode(batch)
        z = output["emb"]
        rows = torch.arange(z.shape[0], device=z.device)
        endpoints = z[rows[:, None], batch["endpoint_indices"]]
        histories = z[rows[:, None, None], batch["history_indices"]]
        p = self.model.predictor
        controls = batch["action"][:, :-1].unflatten(-1, (p.action_block, p.dynamics.action_dim))
        if not torch.isfinite(controls).all():
            raise ValueError("Non-finite supervised primitive forces")
        predicted = p.dynamics.rollout(histories.flatten(-2), controls)[
            ..., -1, : p.observation_dim
        ]
        output["pred_loss"] = (predicted - endpoints[:, 1:]).square().mean()
        output["sigreg_loss"] = self.sigreg(endpoints.transpose(0, 1))
        output["loss"] = output["pred_loss"] + cfg.loss.sigreg.weight * output["sigreg_loss"]
        self.log_dict(
            {f"{stage}/{k}": v.detach() for k, v in output.items() if "loss" in k},
            on_step=True,
            sync_dist=True,
            batch_size=z.shape[0],
        )
        return output
    if cfg.model._target_ in LEWM_TARGETS:
        from .runtime import upstream_module

        return upstream_module("train.lewm").lejepa_forward(self, batch, stage, cfg)

    emb = self.model.encode(batch)["emb"]
    m, k = self.model.predictor.action_dim, self.model.action_block
    controls = batch["action"][:, :-1].unflatten(-1, (k, m))
    if not torch.isfinite(controls).all():
        raise ValueError("Non-finite supervised actions; do not silently replace them")
    loss, parts = world_model_loss(
        self.model.predictor, emb, controls, self.sigreg, cfg.loss.sigreg_weight
    )
    self.log_dict(
        {
            f"{stage}/loss": loss.detach(),
            **{f"{stage}/{key}": value for key, value in parts.items()},
        },
        on_step=stage == "fit",
        on_epoch=True,
        batch_size=emb.shape[0],
        logger=False,
    )
    return {"loss": loss}


def run(cfg):
    import math
    from functools import partial
    from pathlib import Path
    import hydra
    import lightning as pl
    import stable_pretraining as spt
    import stable_worldmodel as swm
    import numpy as np
    from omegaconf import OmegaConf, open_dict
    from lightning.pytorch.callbacks import ModelCheckpoint
    from stable_worldmodel.data import column_normalizer
    from .runtime import upstream_module, RunDirectory, atomic_json, fingerprint
    from .checkpoints import (
        training_plugins,
        Recovery,
        InferenceCheckpoint,
        TrainingHealth,
        save_weights,
        resume_path,
        metric_logger,
    )
    from torchdata.stateful_dataloader import StatefulDataLoader

    upstream = upstream_module("train.lewm")
    get_img_preprocessor = upstream.get_img_preprocessor

    root = Path(__file__).resolve().parents[2]
    checkpoint_dir = root / cfg.checkpoint_dir / cfg.output_model_name
    resume_from = resume_path(checkpoint_dir, cfg.get("resume_from"))
    requested = OmegaConf.to_container(cfg, resolve=False)
    run_owner = RunDirectory(checkpoint_dir, requested, resume=bool(resume_from))

    from .world_model import LEWM_TARGETS

    lewm_mode = cfg.model._target_ in LEWM_TARGETS
    proprio_mode = cfg.model._target_ == "fourier.world_model.ProprioLeWM"
    clip_value = cfg.trainer.gradient_clip_val if lewm_mode else 1.0
    pl.seed_everything(cfg.seed, workers=True)
    torch.set_float32_matmul_precision("highest")
    dataset_args = OmegaConf.to_container(cfg.data.dataset, resolve=True)
    name = dataset_args.pop("name")
    action_spec = cfg.get("action_spec")
    if action_spec:
        from .data import command_dataset

        dataset = command_dataset(name, OmegaConf.to_container(action_spec), **dataset_args)
    else:
        dataset = swm.data.load_dataset(name, **dataset_args)
    m = dataset.get_dim("action")
    action_norm = column_normalizer(dataset, "action", "action")
    # SWM's normalizer fits finite raw train actions. Store the same population stats
    # for the solver's train/eval coordinate conversion (no action encoder).
    actions = np.asarray(dataset.get_col_data("action"))
    actions = actions[~np.isnan(actions).any(axis=1)]
    mean = actions.mean(axis=0)
    scale = np.maximum(actions.std(axis=0), 1e-8)
    transforms = [get_img_preprocessor("pixels", "pixels", cfg.img_size), action_norm]
    if not action_spec:
        transforms.extend(
            column_normalizer(dataset, key, key)
            for key in cfg.data.dataset.keys_to_load
            if key != "action"
            and not key.startswith("pixels")
            and not (proprio_mode and key == "proprio")
        )
    transform = spt.data.transforms.Compose(*transforms)
    dataset.transform = transform
    sampling_generator = torch.Generator().manual_seed(cfg.seed)
    if cfg.validation_dataset:
        val_args = dict(dataset_args)
        val_dataset = (
            command_dataset(cfg.validation_dataset, OmegaConf.to_container(action_spec), **val_args)
            if action_spec
            else swm.data.load_dataset(cfg.validation_dataset, **val_args)
        )
        val_dataset.transform = transform
        train_dataset = dataset
    else:
        train_dataset, val_dataset = spt.data.random_split(
            dataset, [cfg.train_split, 1 - cfg.train_split], generator=sampling_generator
        )
    with open_dict(cfg):
        if proprio_mode:
            values = np.asarray(dataset.get_col_data("proprio"))
            if not np.isfinite(values).all():
                raise ValueError("Nonfinite training proprioception")
            cfg.model.proprio_mean = values.mean(0).tolist()
            cfg.model.proprio_scale = np.maximum(values.std(0), 1e-8).tolist()
        if lewm_mode:
            # Identity accepts this width; the Fourier adapter splits it into raw actions.
            cfg.model.action_encoder.input_dim = cfg.data.dataset.frameskip * m
        else:
            cfg.model.predictor.action_dim = m
            cfg.model.action_mean = mean.tolist()
            cfg.model.action_scale = scale.tolist()
    loader_cfg = OmegaConf.to_container(cfg.loader, resolve=True)
    if loader_cfg["batch_size"] < 2:
        raise ValueError("SIGReg requires batch_size >= 2")
    if loader_cfg["num_workers"] == 0:
        loader_cfg["persistent_workers"] = False
        loader_cfg["prefetch_factor"] = None
    train = StatefulDataLoader(train_dataset, **loader_cfg, generator=sampling_generator)
    val_cfg = {**loader_cfg, "shuffle": False, "drop_last": not lewm_mode}
    val = torch.utils.data.DataLoader(val_dataset, **val_cfg)
    model = hydra.utils.instantiate(cfg.model)
    n_batches = len(train)
    limit = cfg.trainer.get("limit_train_batches", 1.0)
    if isinstance(limit, int):
        n_batches = min(n_batches, limit)
    elif limit < 1:
        n_batches = max(1, int(n_batches * limit))
    total_steps = cfg.trainer.max_epochs * n_batches
    if cfg.trainer.get("max_steps", -1) > 0:
        total_steps = min(total_steps, cfg.trainer.max_steps)
    module = spt.Module(
        model=model,
        sigreg=(
            swm.wm.loss.SIGReg(**cfg.loss.sigreg.kwargs)
            if lewm_mode
            else SIGReg(cfg.loss.knots, cfg.loss.num_proj)
        ),
        forward=partial(swm_forward, cfg=cfg),
        optim={
            "model_opt": {
                "modules": "model",
                "optimizer": dict(cfg.optimizer),
                "gradient_clip_val": clip_value,
                "scheduler": {
                    "type": "LinearWarmupCosineAnnealingLR",
                    "warmup_steps": max(1, int(0.01 * total_steps)),
                    "max_steps": max(2, total_steps),
                },
                "interval": "epoch",
            }
        },
    )
    # These describe how to reload/resume the checkpoint; they are not logs.
    if not (checkpoint_dir / "training_config.yaml").exists():
        OmegaConf.save(cfg, checkpoint_dir / "training_config.yaml")
    metadata = {
        "train_dataset": str(name),
        "validation_dataset": cfg.validation_dataset,
        "train_clips": len(train_dataset),
        "validation_clips": len(val_dataset),
        "primitive_action_dim": m,
        "action_block": cfg.data.dataset.frameskip,
        "action_mean": mean.tolist(),
        "action_scale": scale.tolist(),
        "batch_size": loader_cfg["batch_size"],
        "planned_epochs": cfg.trainer.max_epochs,
        "train_batches_per_epoch": n_batches,
        "planned_max_steps": cfg.trainer.get("max_steps", -1),
        "checkpoint_dir": str(checkpoint_dir),
        "action_spec": OmegaConf.to_container(action_spec) if action_spec else None,
        "dataset_identity": cfg.get("dataset_identity"),
        "model_family": cfg.get("model_family", "fourier"),
        "seed": cfg.seed,
        "benchmark": OmegaConf.to_container(cfg.benchmark, resolve=True)
        if cfg.get("benchmark")
        else None,
        "split_policy": "external_validation"
        if cfg.validation_dataset
        else "upstream_random_clip_split",
        "train_split": cfg.train_split,
    }
    metadata["image_size"] = cfg.img_size
    metadata["prediction_context_frames"] = (
        getattr(model.predictor, "num_frames", 3)
        if hasattr(model.predictor, "context_state") or cfg.get("model_family") == "lewm"
        else 1
    )
    if cfg.get("observation_keys") is not None:
        metadata["observation_keys"] = list(cfg.observation_keys)
    if hasattr(model.predictor, "context_state"):
        metadata["history_encoding"] = model.predictor.history_encoding
        metadata["temporal_state"] = (
            "concatenated image latents, newest to oldest; repeated earliest frame when history is unavailable"
            if model.predictor.history_encoding == "concatenate"
            else "legacy current image latent and backward differences"
        )
        metadata["history_size"] = model.predictor.num_frames
        metadata["history_stride_physical_steps"] = getattr(
            model.predictor, "history_stride_physical_steps", cfg.data.dataset.frameskip
        )
        metadata["history_update"] = (
            "exact primitive shift"
            if metadata["history_stride_physical_steps"] == 1
            else "legacy free stacked-state transition"
        )
    if cfg.get("model_family", "fourier") == "fourier":
        metadata["dynamics_type"] = getattr(model.predictor, "dynamics_type", "fourier")
        metadata["dynamics_parameters"] = sum(p.numel() for p in model.predictor.parameters())
    if proprio_mode:
        metadata["observation_keys"] = ["pixels", "proprio"]
        metadata["proprio_normalization"] = {
            "mean": list(cfg.model.proprio_mean),
            "scale": list(cfg.model.proprio_scale),
            "applied": "inside checkpoint encoder; training artifact only",
        }
        metadata["observation_adapter"] = (
            "linear proprioception fusion before unchanged LeWM projector"
        )
    normalization = {"mean": mean.tolist(), "scale": scale.tolist()}
    if proprio_mode:
        normalization["proprio"] = metadata["proprio_normalization"]
    metadata["normalization_identity"] = fingerprint(normalization)
    atomic_json(checkpoint_dir / "training_metadata.json", metadata)

    class CheckpointMetadata(pl.Callback):
        def on_save_checkpoint(self, trainer, pl_module, checkpoint):
            checkpoint["fourier_training"] = {
                **metadata,
                "global_step": trainer.global_step,
                "completed_full_epochs": trainer.global_step // len(train),
                "config": OmegaConf.to_container(cfg, resolve=True),
            }

        def on_validation_epoch_end(self, trainer, pl_module):
            values = [
                float(v.detach().cpu())
                for v in trainer.callback_metrics.values()
                if torch.is_tensor(v) and v.numel() == 1
            ]
            if not all(math.isfinite(v) for v in values):
                raise ValueError("Non-finite validation metrics")

    trainer_args = OmegaConf.to_container(cfg.trainer, resolve=True)
    # Installed stable-pretraining clips manual optimizers; Lightning requires None here.
    trainer_args["gradient_clip_val"] = None
    recovery = Recovery(checkpoint_dir)
    trainer = pl.Trainer(
        **trainer_args,
        default_root_dir=str(checkpoint_dir),
        logger=metric_logger(checkpoint_dir),
        plugins=training_plugins(),
        callbacks=[
            InferenceCheckpoint(checkpoint_dir, cfg.model),
            CheckpointMetadata(),
            TrainingHealth(checkpoint_dir),
            recovery,
            ModelCheckpoint(
                dirpath=str(checkpoint_dir / "resume"),
                filename="step-{step:09d}",
                every_n_train_steps=cfg.checkpoint_every_steps,
                save_top_k=1,
                save_last=True,
            ),
        ],
        enable_checkpointing=True,
        num_sanity_val_steps=1,
    )
    # stable-pretraining auto-registers an environment-report writer even with
    # logger=False. Drop that file-writing callback before Trainer.fit setup.
    trainer.callbacks = [
        callback
        for callback in trainer.callbacks
        if type(callback).__name__ != "EnvironmentDumpCallback"
    ]
    trainer.gradient_clip_val_ = clip_value
    trainer.gradient_clip_algorithm_ = "norm"
    trainer.fit(
        module,
        train_dataloaders=train,
        val_dataloaders=val,
        ckpt_path=resume_from,
        weights_only=False,
    )
    if recovery.interrupted:
        run_owner.close()
        raise SystemExit(75)  # prevents an afterok evaluation of an incomplete run
    if cfg.trainer.get("max_steps", -1) > 0 and trainer.global_step % n_batches:
        trainer.validate(module, dataloaders=val, verbose=False)
    save_weights(model, checkpoint_dir, cfg.model)
    trainer.save_checkpoint(checkpoint_dir / "resume/last.ckpt")
    atomic_json(
        checkpoint_dir / "status.json", {"status": "completed", "step": trainer.global_step}
    )
    run_owner.close()
    print("CHECKPOINT_SAVED", str(checkpoint_dir), flush=True)


if __name__ == "__main__":
    from pathlib import Path
    import hydra
    from .runtime import CONFIG_ROOT, swm_root
    import os

    os.environ["FOURIER_CONFIG_ROOT"] = str(CONFIG_ROOT)
    os.environ["FOURIER_SWM_ROOT"] = str(swm_root())
    hydra.main(version_base=None, config_path=str(CONFIG_ROOT / "train"), config_name="training")(
        run
    )()
