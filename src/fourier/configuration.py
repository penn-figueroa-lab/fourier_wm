"""Compose project presets and the unchanged upstream model configurations."""

import os

from omegaconf import OmegaConf, open_dict

from .runtime import CONFIG_ROOT, load_catalog, swm_root


def benchmark_policy(environment):
    """Highest-priority published CONTROL benchmark, matched by exact setup."""
    catalog = load_catalog()
    for authority in catalog.paper_priority:
        entry = catalog.published[authority]
        if environment in entry.environments:
            result = OmegaConf.to_container(entry, resolve=True)
            overrides = result.pop("environment_overrides", {})
            result.update(overrides.get(environment, {}))
            result.pop("environments")
            return {"authority": authority, "source": catalog.papers[authority], **result}
    return {
        "authority": "project",
        "protocol": "dataset_goal",
        "train_models": ["fourier", "lewm", "dinowm"],
        "epochs": 10,
        "history": 3,
        "frameskip": 5,
        "unresolved": {},
    }


def compose(name="experiment", overrides=()):
    from hydra import compose as hydra_compose, initialize_config_dir

    # SPT owns the resolver used by its unchanged source configurations.
    # Import before registering fallbacks: SPT registers it unconditionally.
    import stable_pretraining  # noqa: F401

    os.environ["FOURIER_CONFIG_ROOT"] = str(CONFIG_ROOT)
    os.environ["FOURIER_SWM_ROOT"] = str(swm_root())
    if name != "experiment":
        with initialize_config_dir(version_base=None, config_dir=str(CONFIG_ROOT)):
            return hydra_compose(config_name=name, overrides=list(overrides))
    catalog = load_catalog()
    selectors, options = {}, []
    for item in overrides:
        key, _, value = item.partition("=")
        if key in {"environment", "dataset", "protocol", "planner", "machine"}:
            selectors[key] = value
        else:
            options.append(item)
    environment = selectors.get("environment", "pusht")
    if environment not in catalog.environment:
        policy = benchmark_policy(environment)
        if policy["authority"] != "project":
            raise NotImplementedError(
                f"{environment} has a {policy['authority']} benchmark, but its native environment/action/data adapter is not validated here. Do not substitute another environment with a similar name."
            )
        raise ValueError(f"Unknown environment {environment!r}; choose {list(catalog.environment)}")
    if selectors.get("dataset", environment) != environment:
        raise ValueError(
            "Dataset/environment shortcuts must match; use dataset.manifest for the actual artifact"
        )
    if selectors.get("machine", "local") != "local":
        raise ValueError("Site machine profiles belong in an external --config file")
    policy = benchmark_policy(environment)
    protocol = selectors.get("protocol", policy["protocol"])
    planner = selectors.get("planner", "cem")
    cfg = OmegaConf.load(CONFIG_ROOT / "experiment.yaml")
    cfg.environment = catalog.environment[environment]
    cfg.benchmark = policy
    cfg.training_profile = (
        policy["authority"] + "_paper" if policy["authority"] != "project" else "project"
    )
    cfg.dataset.id = cfg.environment.dataset
    if environment == "pointmaze":
        cfg.dataset.regime = "official_navigate"
    cfg.protocol = catalog.protocol[protocol]
    special = cfg.protocol.pop("environment_overrides", {})
    cfg.protocol = OmegaConf.merge(cfg.protocol, special.get(environment, {}))
    cfg.planner = catalog.planner[planner]
    if protocol == "lewm_paper" and environment != "pusht" and planner == "cem":
        cfg.planner.n_steps = 10
    if environment == "dinowm_pointmaze":
        # The image-only adaptation retains LeWM's causal, block-spaced history.
        # The DINO release's proprioceptive/newest-frame protocol is different.
        cfg.history_len = policy["history"]
        cfg.training_profile = "source"
        cfg.dataset.regime = "dinowm_release"
        if planner == "cem":
            cfg.planner.n_steps = 10
    if cfg.dataset.id == "custom":
        cfg.training_profile = "project"
    OmegaConf.set_struct(cfg, True)
    for item in options:
        extra = item.startswith("+")
        key, _, value = item.lstrip("+").partition("=")
        parsed = OmegaConf.from_dotlist(["value=" + value]).value
        OmegaConf.update(cfg, key, parsed, merge=True, force_add=extra)
    return cfg


def training_config(experiment):
    from .data import read_manifest, environment_identity
    from hydra import compose as hydra_compose, initialize_config_dir
    from pathlib import Path

    manifest = read_manifest(experiment.dataset.manifest)
    environment = OmegaConf.to_container(experiment.environment, resolve=True)
    if (
        environment.get("observation_keys") == ["pixels"]
        and environment["family"] == "pointmaze"
        and environment["kwargs"].get("ob_type") != "pixels"
    ):
        raise ValueError(
            "Image-only PointMaze requires native pixel rendering without the task-goal marker"
        )
    if environment_identity(manifest["environment"]) != environment_identity(environment):
        raise ValueError("Training environment and dataset manifest differ")
    if manifest["split"] not in {"train", "expert_train", "project_train"}:
        raise ValueError("Training requires a training split manifest")
    if environment["name"] == "pointmaze" and experiment.dataset.regime == "official_navigate":
        expected_source = (
            "https://rail.eecs.berkeley.edu/datasets/ogbench/pointmaze-large-navigate-v0.npz"
        )
        expected_revision = (
            "sha256:82a73ed8de90ad2b8bf89069253e961918438d127d7f7c7f09c81e642d0d2c61"
        )
        if (
            manifest["source"] != expected_source
            or manifest["revision"] != expected_revision
            or manifest["episodes"] != 1000
            or set(manifest["lengths"]) != {1001}
        ):
            raise ValueError(
                "PointMaze defaults require the complete official OGBench navigate export (1000 x 1001 stored observations). Old waypoint data cannot substitute. Use dataset.regime=project_custom only for explicitly labeled collection studies or smoke checks."
            )
    if environment["name"] == "dinowm_pointmaze" and experiment.dataset.regime == "dinowm_release":
        from .data import require_paper_dataset

        require_paper_dataset(manifest, "dinowm_pointmaze")
        if manifest["episodes"] != 1800 or not experiment.dataset.validation_manifest:
            raise ValueError(
                "DINO-WM Maze requires the original1800/200 episode split and its validation manifest"
            )
    profile = experiment.get("training_profile", "source")
    if profile not in {"swm_paper", "lewm_paper", "dinowm_paper", "source", "project"}:
        raise ValueError("Choose an authority's paper profile, source, or project")
    policy = benchmark_policy(experiment.environment.name)
    paper = profile.endswith("_paper")
    if paper:
        if profile != policy["authority"] + "_paper":
            raise ValueError(
                f"Paper priority selects {policy['authority']} for this setup; use its profile or label an alternative training_profile=project"
            )
        if experiment.dataset.frameskip != policy.get("frameskip", 5):
            raise ValueError("Action blocking differs from this paper benchmark")
        if experiment.dataset.validation_manifest and policy["authority"] != "dinowm":
            raise ValueError(
                "External validation changes the source paper split; use training_profile=project"
            )
        if policy["unresolved"].get(experiment.model_family):
            raise ValueError(policy["unresolved"][experiment.model_family])
        from .data import require_paper_dataset

        require_paper_dataset(manifest, experiment.environment.name)
    family = experiment.model_family
    force_maze = environment["name"] == "dinowm_pointmaze"
    if force_maze and environment.get("observation_keys") != ["pixels"]:
        raise ValueError(
            "D4RL training is image-only. ProprioLeWM remains a legacy checkpoint loader, not this benchmark."
        )
    dynamics_type = experiment.get("dynamics_type", "fourier")
    if dynamics_type not in {"fourier", "linear", "bilinear"} or (
        family != "fourier" and dynamics_type != "fourier"
    ):
        raise ValueError(
            "dynamics_type=linear|bilinear is an ablation of model_family=fourier only"
        )
    overrides = list(experiment.training_overrides)
    if family == "fourier":
        if (
            any(kind in {"force", "torque"} for kind in environment["action"]["kinds"])
            and not force_maze
        ):
            raise ValueError(
                "Native force/torque coordinates are supported, but this training entrypoint uses a single-frame Fourier state. A validated motion-history predictor is required for inertial benchmarks; do not relabel force as velocity."
            )
        cfg = compose("train/training", overrides)
        with open_dict(cfg):
            cfg.model.predictor.dynamics_type = dynamics_type
            if force_maze:
                cfg.model._target_ = "fourier.world_model.AdjacentHistoryWorldModel"
                cfg.model.encoder_checkpointing = True
                cfg.model.predictor._target_ = "fourier.world_model.AdjacentHistoryPredictor"
        if cfg.wm.num_preds != 1:
            raise ValueError("Fourier predicts one action block per endpoint: set wm.num_preds=1")
    elif family in {"lewm", "dinowm"}:
        with initialize_config_dir(
            version_base=None, config_dir=str(swm_root() / "scripts/train/config")
        ):
            cfg = hydra_compose(
                config_name="lewm" if family == "lewm" else "prejepa", overrides=overrides
            )
    else:
        raise ValueError("model_family must be fourier, lewm or dinowm")
    keys = [k for k in ("pixels", "action", "proprio", "state") if k in manifest["columns"]]
    if "pixels" not in keys:
        raise ValueError("Vision training requires pixels")
    with open_dict(cfg):
        # One experiment seed controls initialization AND the source clip split.
        # Otherwise LeWM's source seed (3072) silently overrides this CLI's seed.
        cfg.seed = experiment.seed
        explicit = {item.lstrip("+").split("=", 1)[0] for item in overrides}
        if experiment.machine.num_workers is not None and "num_workers" not in explicit:
            cfg.num_workers = experiment.machine.num_workers
        if "trainer.accelerator" not in explicit:
            cfg.trainer.accelerator = (
                "gpu" if experiment.machine.device == "cuda" else experiment.machine.device
            )
        cfg.training_profile = profile
        cfg.benchmark = {**policy, "paper_training": paper, "selected_profile": profile}
        if profile == "project" and "trainer.max_epochs" not in explicit:
            cfg.trainer.max_epochs = policy["epochs"]
        observation_keys = environment.get("observation_keys")
        if observation_keys is not None:
            cfg.observation_keys = list(observation_keys)
            if observation_keys != ["pixels"]:
                raise ValueError("This benchmark supports image-only model observations")
            elif family == "dinowm":
                for override in overrides:
                    key, _, value = override.lstrip("+").partition("=")
                    if key.startswith("wm.encoding.") and key != "wm.encoding.action":
                        raise ValueError(
                            "Image-only benchmarks cannot encode proprioception or state"
                        )
                    if key == "wm.encoding" and set(
                        OmegaConf.from_dotlist(["encoding=" + value]).encoding
                    ) != {"action"}:
                        raise ValueError(
                            "Image-only benchmarks cannot encode proprioception or state"
                        )
                cfg.wm.encoding = {"action": cfg.wm.encoding.action}
            else:
                keys = ["pixels", "action"]
        if paper:
            if any(
                x.lstrip("+").startswith("trainer.max_epochs=")
                and x.split("=", 1)[1] != str(policy["epochs"])
                for x in overrides
            ):
                raise ValueError(
                    f"This paper training budget is {policy['epochs']} epochs; use training_profile=project for a different experiment"
                )
            cfg.trainer.max_epochs = policy["epochs"]
            history = policy["history"]
            if any(
                x.lstrip("+").startswith("wm.history_size=") and x.split("=", 1)[1] != str(history)
                for x in overrides
            ):
                raise ValueError(f"Paper history is {history} for {experiment.environment.name}")
            cfg.wm.history_size = history
            if policy["authority"] == "lewm" and family == "dinowm":
                cfg.wm.encoding = {"action": 10}  # LeWM paper's pixels-only DINO-WM baseline.
        if force_maze and cfg.wm.history_size != policy["history"]:
            raise ValueError(
                "Image-only D4RL comparison requires LeWM history_size=3 for both models"
            )
        cfg.model_family = family
        cfg.dataset_identity = manifest["identity"]
        cfg.dataset_manifest = str(Path(experiment.dataset.manifest).resolve())
        cfg.checkpoint_dir = str(Path(experiment.machine.runs_root).expanduser().resolve())
        cfg.output_model_name = experiment.run_name
        cfg.resume_from = experiment.resume_from
        cfg.checkpoint_every_steps = 1000
        cfg.validation_dataset = None
        cfg.subdir = "."
        if experiment.dataset.validation_manifest:
            val = read_manifest(experiment.dataset.validation_manifest)
            if (
                environment_identity(val["environment"]) != environment_identity(environment)
                or val["sha256"] == manifest["sha256"]
                or val["split"] not in {"validation", "val", "expert_val", "project_val"}
            ):
                raise ValueError("Invalid validation manifest")
            cfg.validation_dataset = val["path"]
            cfg.validation_identity = val["identity"]
            if (
                environment["name"] == "dinowm_pointmaze"
                and experiment.dataset.regime == "dinowm_release"
            ):
                require_paper_dataset(val, "dinowm_pointmaze")
                if val["episodes"] != 200:
                    raise ValueError(
                        "DINO-WM Maze validation must contain the original200 held-out episodes"
                    )
        if family == "dinowm":
            cfg.dataset_name = manifest["path"]
            cfg.frameskip = experiment.dataset.frameskip
        else:
            cfg.data.dataset = {
                "name": manifest["path"],
                "num_steps": "${eval:'${wm.num_preds} + ${wm.history_size}'}",
                "frameskip": experiment.dataset.frameskip,
                "keys_to_load": keys,
                "keys_to_cache": [k for k in keys if k != "pixels"],
            }
            if force_maze and family == "fourier":
                cfg.data.dataset.history_frames = cfg.wm.history_size
            cfg.action_spec = environment["action"] if family == "fourier" else None
        # Training defaults remain those of the source model unless explicitly overridden.
    if paper:
        expected = {
            "train_split": 0.9,
            "wm.num_preds": 1,
            "model.predictor.num_frames": policy["history"],
        }
        if family == "dinowm":
            expected.update(
                {
                    "image_size": 224,
                    "batch_size": 32,
                    "optimizer.type": "AdamW",
                    "optimizer.lr": 5e-4,
                    "optimizer.weight_decay": 0.0,
                    "model.encoder.name": "dinov2_small",
                }
            )
        else:
            expected.update(
                {
                    "img_size": 224,
                    "patch_size": 14,
                    "encoder_scale": "tiny",
                    "embed_dim": 192,
                    "loader.batch_size": 128,
                    "optimizer.type": "AdamW",
                    "optimizer.lr": 5e-5,
                    "optimizer.weight_decay": 1e-3,
                    "loss.sigreg.weight": 0.09,
                    "loss.sigreg.kwargs.num_proj": 1024,
                    "loss.sigreg.kwargs.knots": 17,
                }
            )
            expected.update(
                {
                    "model.encoder._target_": "stable_pretraining.backbone.utils.vit_hf",
                    "model.encoder.size": "tiny",
                    "model.encoder.patch_size": 14,
                    "model.encoder.image_size": 224,
                    "model.encoder.pretrained": False,
                    "model.projector.hidden_dim": 2048,
                }
            )
        if family != "fourier":
            expected.update(
                {
                    "model.predictor.depth": 6,
                    "model.predictor.heads": 16,
                    "model.predictor.mlp_dim": 2048,
                    "model.predictor.dim_head": 64,
                    "model.predictor.dropout": 0.1,
                    "model.predictor.emb_dropout": 0.0,
                }
            )
        expected.update(
            {
                "trainer.max_steps": -1,
                "trainer.limit_train_batches": 1.0,
                "trainer.limit_val_batches": 1.0,
            }
        )
        changed = [
            key
            for key, value in expected.items()
            if OmegaConf.select(cfg, key, default=value if key.startswith("trainer.") else None)
            != value
        ]
        if changed:
            raise ValueError(
                f"Modified paper training fields {changed}; use training_profile=project for smoke tests or ablations"
            )
    return cfg
