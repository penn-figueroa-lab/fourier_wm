"""Endpoint prediction + optional SIGReg, independent of encoder/data format."""
import torch
from torch import nn


class SIGReg(nn.Module):
    """LeWM's single-device sketch Gaussian regularizer on [T, B, D].

    Adapted from LeWM module.py (Lucas Maes, MIT; see THIRD_PARTY_LICENSES).
    This is not a guarantee against all representation collapse.
    """

    def __init__(self, knots=17, num_proj=1024):
        super().__init__()
        if knots < 2 or num_proj < 1:
            raise ValueError('Require knots>=2 and num_proj>=1')
        self.num_proj = num_proj
        t = torch.linspace(0, 3, knots)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt)
        weights[[0, -1]] = dt
        self.register_buffer('t', t)
        self.register_buffer('phi', torch.exp(-t.square() / 2))
        self.register_buffer('weights', weights * self.phi)

    def forward(self, embeddings):
        if embeddings.ndim != 3 or embeddings.shape[1] < 2:
            raise ValueError('Expected [T, B>=2, D]')
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
    if (embeddings.ndim != 3 or embeddings.shape[1] < 2 or controls.ndim != 4
            or controls.shape[:2] != (embeddings.shape[0], embeddings.shape[1] - 1)):
        raise ValueError('Expected [B,T>=2,D] and [B,T-1,H,m]')
    predicted = dynamics.rollout(embeddings[:, :-1], controls)[..., -1, :]
    return (predicted - embeddings[:, 1:]).square().mean()


def world_model_loss(dynamics, embeddings, controls, regularizer, sigreg_weight=.09):
    """Train any encoder/projector jointly with Fourier dynamics."""
    if sigreg_weight < 0:
        raise ValueError('SIGReg weight must be nonnegative')
    forward = endpoint_loss(dynamics, embeddings, controls)
    regularization = regularizer(embeddings.transpose(0, 1))
    loss = forward + sigreg_weight * regularization
    return loss, {'forward': forward.detach(), 'sigreg': regularization.detach()}


# Project-owned integration glue. The definitions above are the supplied code.
def swm_forward(self, batch, stage, cfg):
    emb = self.model.encode(batch)["emb"]
    m, k = self.model.predictor.action_dim, self.model.action_block
    controls = batch["action"][:, :-1].unflatten(-1, (k, m))
    if not torch.isfinite(controls).all():
        raise ValueError("Non-finite supervised actions; do not silently replace them")
    loss, parts = world_model_loss(
        self.model.predictor, emb, controls, self.sigreg, cfg.loss.sigreg_weight)
    self.log_dict({f"{stage}/loss": loss.detach(),
                   **{f"{stage}/{key}": value for key, value in parts.items()}},
                  on_step=stage == "fit", on_epoch=True, batch_size=emb.shape[0])
    return {"loss": loss}


def run(cfg):
    import json
    import math
    import os
    import time
    from functools import partial
    from pathlib import Path
    import hydra
    import lightning as pl
    import stable_pretraining as spt
    import stable_worldmodel as swm
    import numpy as np
    from omegaconf import OmegaConf, open_dict
    from lightning.pytorch.loggers import CSVLogger
    from lightning.pytorch.callbacks import ModelCheckpoint
    from stable_worldmodel.data import column_normalizer
    from stable_worldmodel.wm.utils import save_pretrained
    from scripts.train.lewm import get_img_preprocessor, SaveCkptCallback

    pl.seed_everything(cfg.seed, workers=True)
    torch.set_float32_matmul_precision("highest")
    dataset_args = OmegaConf.to_container(cfg.data.dataset, resolve=True)
    name = dataset_args.pop("name")
    dataset = swm.data.load_dataset(name, **dataset_args)
    m = dataset.get_dim("action")
    action_norm = column_normalizer(dataset, "action", "action")
    # SWM's normalizer fits finite raw train actions. Store the same population stats
    # for the solver's train/eval coordinate conversion (no action encoder).
    actions = np.asarray(dataset.get_col_data("action"))
    actions = actions[~np.isnan(actions).any(axis=1)]
    mean = actions.mean(axis=0)
    scale = np.maximum(actions.std(axis=0), 1e-8)
    transform = spt.data.transforms.Compose(
        get_img_preprocessor("pixels", "pixels", cfg.img_size), action_norm)
    dataset.transform = transform
    if cfg.validation_dataset:
        val_args = dict(dataset_args)
        val_dataset = swm.data.load_dataset(cfg.validation_dataset, **val_args)
        val_dataset.transform = transform
        train_dataset = dataset
    else:
        train_dataset, val_dataset = spt.data.random_split(
            dataset, [cfg.train_split, 1-cfg.train_split],
            generator=torch.Generator().manual_seed(cfg.seed))
    with open_dict(cfg):
        cfg.model.predictor.action_dim = m
        cfg.model.action_mean = mean.tolist()
        cfg.model.action_scale = scale.tolist()
    loader_cfg = OmegaConf.to_container(cfg.loader, resolve=True)
    if loader_cfg["batch_size"] < 2:
        raise ValueError("SIGReg requires batch_size >= 2")
    if loader_cfg["num_workers"] == 0:
        loader_cfg["persistent_workers"] = False
    train = torch.utils.data.DataLoader(
        train_dataset, **loader_cfg, generator=torch.Generator().manual_seed(cfg.seed))
    val_cfg = {**loader_cfg, "shuffle": False, "drop_last": True}
    val = torch.utils.data.DataLoader(val_dataset, **val_cfg)
    model = hydra.utils.instantiate(cfg.model)
    n_batches = len(train)
    limit = cfg.trainer.limit_train_batches
    if isinstance(limit, int):
        n_batches = min(n_batches, limit)
    elif limit < 1:
        n_batches = max(1, int(n_batches * limit))
    total_steps = cfg.trainer.max_epochs * n_batches
    if cfg.trainer.get("max_steps", -1) > 0:
        total_steps = min(total_steps, cfg.trainer.max_steps)
    module = spt.Module(
        model=model, sigreg=SIGReg(cfg.loss.knots, cfg.loss.num_proj),
        forward=partial(swm_forward, cfg=cfg),
        optim={"model_opt": {
            "modules": "model", "optimizer": dict(cfg.optimizer),
            "gradient_clip_val": 1.0,
            "scheduler": {"type": "LinearWarmupCosineAnnealingLR",
                          "warmup_steps": max(1, int(.01 * total_steps)),
                          "max_steps": max(2, total_steps)},
            "interval": "epoch"}})
    root = Path(__file__).resolve().parents[2]
    run_dir = root / "artifacts" / "training" / cfg.output_model_name
    checkpoint_dir = root / cfg.checkpoint_dir / cfg.output_model_name
    resume_from = cfg.get("resume_from")
    run_dir.mkdir(parents=True, exist_ok=bool(resume_from))
    checkpoint_dir.mkdir(parents=True, exist_ok=bool(resume_from))
    OmegaConf.save(cfg, run_dir / "config.yaml")
    record = {
        "train_dataset": str(name), "validation_dataset": cfg.validation_dataset,
        "train_clips": len(train_dataset), "validation_clips": len(val_dataset),
        "primitive_action_dim": m, "action_block": model.action_block,
        "action_mean": mean.tolist(), "action_scale": scale.tolist(),
        "batch_size": loader_cfg["batch_size"], "planned_epochs": cfg.trainer.max_epochs,
        "train_batches_per_epoch": n_batches, "planned_max_steps": cfg.trainer.get("max_steps", -1), "pid": os.getpid(),
        "torch": torch.__version__, "status": "running",
        "checkpoint_dir": str(checkpoint_dir),
    }
    (run_dir / "run.json").write_text(json.dumps(record, indent=2))

    class Metrics(pl.Callback):
        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            if trainer.global_step % cfg.checkpoint_every_steps == 0:
                record.update(global_step=trainer.global_step,
                              completed_full_epochs=trainer.global_step // len(train),
                              fractional_epochs=trainer.global_step / len(train),
                              elapsed_seconds=time.time()-start)
                (run_dir / "run.json").write_text(json.dumps(record, indent=2))

        def on_validation_epoch_end(self, trainer, pl_module):
            values = {k: float(v.detach().cpu()) for k,v in trainer.callback_metrics.items()
                      if torch.is_tensor(v) and v.numel() == 1}
            if not all(math.isfinite(v) for v in values.values()):
                raise ValueError(f"Non-finite metrics: {values}")
            if not trainer.sanity_checking:
                with (run_dir / "epochs.jsonl").open("a") as f:
                    f.write(json.dumps({"epoch": trainer.current_epoch + (0 if trainer.state.fn.value == "validate" else 1),
                                        "step": trainer.global_step, **values}) + "\n")

    trainer = pl.Trainer(
        **OmegaConf.to_container(cfg.trainer, resolve=True),
        default_root_dir=str(run_dir),
        logger=CSVLogger(str(run_dir), name="metrics"),
        callbacks=[SaveCkptCallback(str(checkpoint_dir), cfg.model), Metrics(),
                   ModelCheckpoint(dirpath=str(checkpoint_dir / "resume"),
                                   filename="step-{step:09d}",
                                   every_n_train_steps=cfg.checkpoint_every_steps,
                                   save_top_k=1, save_last=True)],
        enable_checkpointing=True, num_sanity_val_steps=1)
    # Installed stable_pretraining supports these manual-optimization aliases.
    # Lightning rejects its native gradient_clip_val for manual optimization.
    trainer.gradient_clip_val_ = 1.0
    trainer.gradient_clip_algorithm_ = "norm"
    start = time.time()
    trainer.fit(module, train_dataloaders=train, val_dataloaders=val,
                ckpt_path=resume_from)
    # A max_steps stop inside an epoch does not trigger Lightning epoch-end validation.
    if cfg.trainer.get("max_steps", -1) > 0 and trainer.global_step % n_batches:
        trainer.validate(module, dataloaders=val, verbose=True)
    save_pretrained(model, str(checkpoint_dir), cfg.model, filename="weights.pt")
    record.update(status="completed", epochs=trainer.current_epoch,
                  global_step=trainer.global_step, completed_full_epochs=trainer.global_step // len(train),
                  fractional_epochs=trainer.global_step / len(train), elapsed_seconds=time.time()-start)
    record["metrics"] = {k: float(v.detach().cpu()) for k,v in trainer.callback_metrics.items()
                         if torch.is_tensor(v) and v.numel() == 1}
    (run_dir / "run.json").write_text(json.dumps(record, indent=2))
    print(json.dumps(record, indent=2), flush=True)


if __name__ == "__main__":
    from pathlib import Path
    import hydra
    hydra.main(version_base=None,
               config_path=str(Path(__file__).resolve().parents[2] / "configs" / "train"),
               config_name="training")(run)()
