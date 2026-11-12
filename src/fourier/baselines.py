"""DINO-WM orchestration using the pinned upstream forward/model definitions."""

from functools import partial
from pathlib import Path

import hydra
import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from omegaconf import OmegaConf, open_dict
from torchdata.stateful_dataloader import StatefulDataLoader

from .checkpoints import (
    training_plugins,
    Recovery,
    InferenceCheckpoint,
    TrainingHealth,
    save_weights,
    resume_path,
    metric_logger,
)
from .runtime import RunDirectory, atomic_json, upstream_module
from .loading import load_dataset


def preserve_extra_batch_axis(module, arguments):
    """Restore a batch of one removed by upstream dinowm_forward.squeeze().

    Training clips always have at least two frames. Native Embedder expects
    [B,T,D]; its parameters and computation are unchanged by this shape adapter.
    """
    value, *rest = arguments
    if value.ndim == 2:
        value = value.unsqueeze(0)
    return (value, *rest)


def train_dinowm(cfg):
    upstream = upstream_module("train.prejepa")
    pl.seed_everything(cfg.seed, workers=True)
    directory = Path(cfg.checkpoint_dir) / cfg.output_model_name
    with RunDirectory(
        directory, OmegaConf.to_container(cfg, resolve=False), resume=bool(cfg.resume_from)
    ):
        keys = list(cfg.wm.encoding)
        dataset = load_dataset(
            cfg.dataset_name,
            num_steps=cfg.n_steps,
            frameskip=cfg.frameskip,
            keys_to_load=["pixels", *keys],
            keys_to_cache=keys,
        )
        dataset.transform = spt.data.transforms.Compose(
            upstream.get_img_preprocessor("pixels", "pixels", cfg.image_size),
            *[swm.data.column_normalizer(dataset, k, k) for k in keys],
        )
        rng = torch.Generator().manual_seed(cfg.seed)
        if cfg.validation_dataset:
            train = dataset
            val = load_dataset(
                cfg.validation_dataset,
                num_steps=cfg.n_steps,
                frameskip=cfg.frameskip,
                keys_to_load=["pixels", *keys],
                keys_to_cache=keys,
            )
            val.transform = dataset.transform
        else:
            train, val = spt.data.random_split(
                dataset, [cfg.train_split, 1 - cfg.train_split], generator=rng
            )
        train = StatefulDataLoader(
            train,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            drop_last=True,
            persistent_workers=cfg.num_workers > 0,
            pin_memory=True,
            shuffle=True,
            generator=rng,
        )
        val = torch.utils.data.DataLoader(
            val, batch_size=cfg.batch_size, num_workers=cfg.num_workers, pin_memory=True
        )
        encoder = hydra.utils.instantiate(cfg.model.encoder).eval().requires_grad_(False)
        is_cnn = hasattr(encoder.config, "hidden_sizes")
        width = encoder.config.hidden_sizes[-1] if is_cnn else encoder.config.hidden_size
        with open_dict(cfg):
            cfg.model.predictor.dim = width + sum(cfg.wm.encoding.values())
            cfg.model.predictor.num_patches = (
                1 if is_cnn else (cfg.image_size // cfg.patch_size) ** 2
            )
            cfg.model.extra_encoders = {
                "_target_": "torch.nn.ModuleDict",
                "modules": {
                    k: {
                        "_target_": "stable_worldmodel.wm.prejepa.module.Embedder",
                        "in_chans": dataset.get_dim(k) * (cfg.frameskip if k == "action" else 1),
                        "emb_dim": int(cfg.wm.encoding[k]),
                    }
                    for k in keys
                },
            }
        model = hydra.utils.instantiate(cfg.model, encoder=encoder)
        for extra_encoder in model.extra_encoders.values():
            extra_encoder.register_forward_pre_hook(preserve_extra_batch_axis)
        module = spt.Module(
            model=model,
            forward=partial(upstream.dinowm_forward, cfg=cfg),
            optim={"model_opt": {"modules": "model", "optimizer": dict(cfg.optimizer)}},
        )
        recovery = Recovery(directory)
        arguments = OmegaConf.to_container(cfg.trainer, resolve=True)
        clip = arguments.pop("gradient_clip_val", None)
        trainer = pl.Trainer(
            **arguments,
            gradient_clip_val=None,
            default_root_dir=str(directory),
            logger=metric_logger(directory),
            plugins=training_plugins(),
            num_sanity_val_steps=1,
            callbacks=[
                InferenceCheckpoint(directory, cfg.model, interval=5),
                TrainingHealth(directory),
                recovery,
                ModelCheckpoint(
                    dirpath=directory / "resume",
                    every_n_train_steps=cfg.checkpoint_every_steps,
                    save_last=True,
                    save_top_k=1,
                ),
            ],
        )
        trainer.gradient_clip_val_, trainer.gradient_clip_algorithm_ = clip, "norm"
        cfg_copy = OmegaConf.to_container(cfg, resolve=True)
        if not (directory / "training_config.json").exists():
            atomic_json(directory / "training_config.json", cfg_copy)
        from .actions import ActionNormalizer

        stats = ActionNormalizer.fit(dataset.get_col_data("action"))
        atomic_json(
            directory / "training_metadata.json",
            {
                "model_family": "dinowm",
                "action_block": cfg.frameskip,
                "action_mean": stats.mean.tolist(),
                "action_scale": stats.scale.tolist(),
                "action_spec": None,
                "dataset_identity": cfg.dataset_identity,
                "image_size": cfg.image_size,
                "seed": cfg.seed,
                "benchmark": OmegaConf.to_container(cfg.benchmark, resolve=True)
                if cfg.get("benchmark")
                else None,
                "split_policy": "external_validation"
                if cfg.validation_dataset
                else "upstream_random_clip_split",
                "train_split": cfg.train_split,
                "backbone_revision": getattr(encoder.config, "_commit_hash", None),
            },
        )
        trainer.fit(
            module,
            train_dataloaders=train,
            val_dataloaders=val,
            ckpt_path=resume_path(directory, cfg.resume_from),
            weights_only=False,
        )
        if not recovery.interrupted:
            trainer.save_checkpoint(directory / "resume/last.ckpt")
            save_weights(model, directory, cfg.model)
            atomic_json(
                directory / "status.json", {"status": "completed", "step": trainer.global_step}
            )
        else:
            raise SystemExit(75)
