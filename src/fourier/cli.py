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
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download and prepare the pinned release in --output",
    )
    parser.add_argument("--episodes", type=int)
    parser.add_argument(
        "--parent-budget", type=int, help="Maximum source parents in one dense replay job"
    )
    parser.add_argument(
        "--resume-data",
        action="store_true",
        help="Resume an interrupted dataset conversion at its last complete episode",
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
    if args.download and args.command != "prepare-data":
        parser.error("--download belongs to prepare-data")
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
            "parent_budget",
            "catalog",
            "archive",
        ):
            if getattr(args, key) is None:
                setattr(args, key, cfg.preparation.get(key))
            cfg.preparation[key] = getattr(args, key)
        args.collect = args.collect or cfg.preparation.collect
        cfg.preparation.collect = args.collect
        args.download = args.download or cfg.preparation.get("download", False)
        cfg.preparation.download = args.download
        if args.download and (
            args.collect
            or args.source
            or args.archive
            or args.source_url
            or args.revision
            or args.expected_sha256
        ):
            parser.error(
                "--download uses the pinned catalog; omit --collect and manual source/archive/provenance options"
            )
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

        if args.download:
            if not args.output:
                parser.error("--download requires an output directory via --output")
            from .download import prepare_download

            print(
                json.dumps(
                    prepare_download(
                        OmegaConf.to_container(cfg.environment, resolve=True),
                        args.output,
                        catalog=args.catalog,
                        split=args.split,
                        resume=args.resume_data,
                        limit=args.episodes,
                    ),
                    indent=2,
                )
            )
            return
        if not args.source or not args.output:
            parser.error("prepare-data requires --source and --output")
        env = OmegaConf.to_container(cfg.environment, resolve=True)
        if env["family"] == "deformable_dense":
            if not args.collect or args.catalog not in {None, "custom"}:
                parser.error(
                    "Dense PyFlex replay requires --collect and the verified native data root in --source"
                )
            from .dense_deformable import prepare_dense

            print(
                json.dumps(
                    prepare_dense(
                        args.source,
                        args.output,
                        env,
                        resume=args.resume_data,
                        limit=args.episodes,
                        parent_budget=args.parent_budget,
                    ),
                    indent=2,
                )
            )
            return
        if env["family"] == "deformable":
            if (
                args.collect
                or args.archive
                or args.source_url
                or args.revision
                or args.expected_sha256
                or args.catalog not in {None, env["dataset"]}
            ):
                parser.error(
                    "Deformable import uses pinned split ZIPs in --source; omit collection/manual provenance options"
                )
            from .deformable import prepare_release
            from .data import dataset_catalog

            if args.episodes is not None or args.split != "train":
                parser.error("Deformable import prepares the complete train/validation split")
            entry = dataset_catalog()[env["dataset"]]
            print(
                json.dumps(
                    prepare_release(
                        entry,
                        env,
                        args.output,
                        env["dataset"],
                        raw=args.source,
                        resume=args.resume_data,
                    ),
                    indent=2,
                )
            )
            return
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
        if cfg.environment.family == "deformable":
            import importlib.util

            report["deformable"] = {
                "dataset_import": "supported without PyFlex",
                "native_action": "complete push [start_x,start_z,end_x,end_z]",
                "fourier_training": "requires dense commanded-motion replay",
                "pyflex_importable": importlib.util.find_spec("pyflex") is not None,
                "simulator_root": os.environ.get("DINO_WM_ROOT"),
                "simulator_gpu_validation": "not established by this doctor check",
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
