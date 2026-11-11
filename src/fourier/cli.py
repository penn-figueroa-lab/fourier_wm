"""One installed command for preparing, training, evaluating and scheduling."""

import argparse
import json
import os
from pathlib import Path
from importlib.metadata import entry_points


def main(argv=None):
    parser = argparse.ArgumentParser(prog="fourier-wm")
    parser.add_argument(
        "command", choices=["prepare-data", "train", "eval", "submit", "doctor", "summarize"]
    )
    parser.add_argument("--config", help="Previously resolved experiment YAML")
    parser.add_argument(
        "--export-config", help="Save the resolved common experiment config and exit"
    )
    parser.add_argument("--environment", help="Environment and matching dataset preset")
    parser.add_argument("--source")
    parser.add_argument("--output")
    parser.add_argument("--source-url")
    parser.add_argument("--revision")
    parser.add_argument("--expected-sha256")
    parser.add_argument("--catalog", help="Pinned paper dataset ID, or custom")
    parser.add_argument("--archive", help="Original author archive for provenance verification")
    parser.add_argument("--split")
    parser.add_argument("--collect", action="store_true")
    parser.add_argument("--episodes", type=int)
    parser.add_argument(
        "--resume-data",
        action="store_true",
        help="Resume an interrupted official PointMaze image export",
    )
    parser.add_argument("--execute", action="store_true", help="Submit a generated Slurm script")
    parser.add_argument(
        "--batch-command", choices=["prepare-data", "train", "eval"], default="train"
    )
    parser.add_argument("--experiments", nargs="+")
    parser.add_argument("--dependency")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--train-option",
        action="append",
        default=[],
        help="Repeatable override of the source training config, e.g. num_workers=0",
    )
    args, overrides = parser.parse_known_args(argv)
    if any("=" not in item for item in overrides):
        parser.error("Config overrides must be key=value")
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    from omegaconf import OmegaConf
    from .configuration import compose, training_config
    from .runtime import atomic_json, swm_root, UPSTREAM_COMMIT, verify_upstream

    if args.config:
        cfg = OmegaConf.load(args.config)
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    else:
        if args.environment:
            overrides = [
                f"environment={args.environment}",
                f"dataset={args.environment}",
                *overrides,
            ]
        cfg = compose(overrides=overrides)
    cfg.training_overrides = [*cfg.training_overrides, *args.train_option]
    if args.command == "prepare-data":
        for key in (
            "source",
            "output",
            "source_url",
            "revision",
            "expected_sha256",
            "split",
            "episodes",
            "catalog",
            "archive",
        ):
            if getattr(args, key) is None:
                setattr(args, key, cfg.preparation.get(key))
            cfg.preparation[key] = getattr(args, key)
        args.collect = args.collect or cfg.preparation.collect
        cfg.preparation.collect = args.collect
    if args.export_config:
        destination = Path(args.export_config).expanduser()
        if destination.exists():
            raise FileExistsError(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(cfg, destination, resolve=True)
        return
    if args.command in {"train", "eval", "prepare-data"}:
        for hook in entry_points(group="fourier_wm.before_command"):
            hook.load()(args, cfg)
    if args.dry_run:
        value = training_config(cfg) if args.command == "train" else cfg
        print(OmegaConf.to_yaml(value, resolve=args.command != "train"))
        return
    if args.command in {"train", "eval", "prepare-data"}:
        verification = verify_upstream()
        if not verification["ok"]:
            raise RuntimeError(
                f"Vendored upstream differs from the pinned tree: {verification['changed']}"
            )
    if args.command in {"train", "eval"}:
        if not cfg.run_name or Path(cfg.run_name).name != cfg.run_name:
            parser.error("Set a unique run_name without path separators")
        if not cfg.dataset.manifest:
            parser.error("Set dataset.manifest to the prepare-data output")
    if args.command == "prepare-data":
        from .data import prepare_manifest

        if not args.source or not args.output:
            parser.error("prepare-data requires --source and --output")
        env = OmegaConf.to_container(cfg.environment, resolve=True)
        if args.catalog == "ogbench_pointmaze_navigate":
            if env["name"] != "pointmaze" or args.collect:
                parser.error(
                    "The official navigate importer requires --environment pointmaze and existing raw NPZs"
                )
            from .data import import_ogbench_pointmaze

            result = import_ogbench_pointmaze(
                args.source, args.output, resume=args.resume_data, limit=args.episodes
            )
            print(json.dumps(result, indent=2))
            return
        if args.collect:
            from .environments import collect

            if args.episodes is not None:
                env["collection"]["episodes"] = args.episodes
            collect(env, args.source, episodes=args.episodes)
            args.source_url = (
                args.source_url or "project:seeded-" + env["collection"]["policy"] + "-policy"
            )
            args.revision = args.revision or UPSTREAM_COMMIT
        catalog = "custom" if args.collect else (args.catalog or cfg.dataset.id)
        if catalog != "custom":
            from .data import dataset_catalog

            entry = dataset_catalog()[catalog]
            args.source_url = args.source_url or entry["source"]
            args.revision = args.revision or entry["revision"]
        result = prepare_manifest(
            args.source,
            args.output,
            env,
            source_url=args.source_url,
            revision=args.revision,
            split=args.split,
            expected_sha256=args.expected_sha256,
            catalog=catalog,
            archive=args.archive,
        )
        print(
            json.dumps(
                {
                    "manifest": args.output,
                    "identity": result["identity"],
                    "episodes": result["episodes"],
                },
                indent=2,
            )
        )
    elif args.command == "train":
        train_cfg = training_config(cfg)
        if cfg.model_family == "dinowm":
            from .baselines import train_dinowm

            train_dinowm(train_cfg)
        else:
            from .training import run

            run(train_cfg)
    elif args.command == "eval":
        from .evaluation import evaluate

        print(json.dumps(evaluate(cfg), indent=2))
    elif args.command == "submit":
        adapters = list(entry_points(group="fourier_wm.submit"))
        if len(adapters) != 1:
            parser.error("Install one external scheduler adapter to use submit")
        print(json.dumps(adapters[0].load()(args, cfg), indent=2))
    elif args.command == "doctor":
        report = {
            "upstream_expected": UPSTREAM_COMMIT,
            "swm_root": str(swm_root()),
            "allocation": os.environ.get("SLURM_JOB_ID"),
            "machine": cfg.machine.name,
            "upstream_verification": verify_upstream(),
        }
        for hook in entry_points(group="fourier_wm.before_command"):
            hook.load()(args, cfg)
        for hook in entry_points(group="fourier_wm.doctor"):
            report.update(hook.load()(args, cfg))
        if cfg.machine.get("allow_compute", True):
            import torch

            report.update(
                torch=torch.__version__,
                cuda=torch.version.cuda,
                gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None,
            )
            if torch.cuda.is_available():
                x = torch.ones((16, 16), device="cuda")
                report["cuda_matmul_ok"] = bool(torch.isfinite(x @ x).all())
                report["cuda_bf16_matmul_ok"] = bool(
                    torch.isfinite(x.bfloat16() @ x.bfloat16()).all()
                )
            if cfg.checkpoint:
                from .checkpoints import inspect_checkpoint

                report["checkpoint"] = inspect_checkpoint(
                    cfg.checkpoint, OmegaConf.to_container(cfg.environment, resolve=True)
                )
        elif cfg.checkpoint:
            raise ValueError(
                "Checkpoint inspection requires a machine profile with allow_compute=true"
            )
        if args.output:
            atomic_json(args.output, report)
        print(json.dumps(report, indent=2))
    elif args.command == "summarize":
        root = Path(cfg.machine.runs_root).expanduser()
        records = [
            {"run": str(p.parent), **json.loads(p.read_text())}
            for p in sorted(root.glob("*/results.json"))
        ]
        if args.output:
            atomic_json(args.output, records)
        print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
