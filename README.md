# Fourier World Model

Fourier dynamics and MPC around a pinned, unchanged [Stable World Model](https://github.com/galilai-group/stable-worldmodel) checkout. LeWM and DINO-WM reuse upstream implementations. Fourier replaces LeWM's transformer and prediction head, removes action embedding, and advances dynamics at each primitive action.

**Testing the shared Two-Room weights? Start with [the Two-Room guide](docs/TWOROOM.md).** It covers checking a checkpoint without data, replaying saved CEM tasks, and testing frequent replanning.

## Install

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then run from this checkout:

```bash
bash scripts/install.sh
source .venv/bin/activate
fourier-wm doctor
export FOURIER_RUNS=/absolute/path/to/experiment-runs
```

Python 3.11 and CUDA 12.8 dependencies are locked in `uv.lock`. Commands find their configurations independently of the working directory. Keep data, weights and outputs outside Git.

## Layout

| Location | Purpose |
|---|---|
| `stable-worldmodel/` | Unmodified upstream source and licenses; revision in `UPSTREAM.json` |
| `src/fourier/predictor.py`, `world_model.py` | Dynamics and LeWM integration |
| `training.py`, `baselines.py`, `checkpoints.py` | Source training adapters, checkpoint loading and recovery |
| `actions.py`, `data.py`, `environments.py` | Command conversion, dataset provenance and faithful resets |
| `control.py`, `sqp.py`, `evaluation.py` | MPC, projected Gauss–Newton, TV-LQR and saved-task evaluation |
| `configs/` | Link to the four packaged YAML files in `src/fourier/configs/` |
| `scripts/` | Installation, dataset-free Two-Room demo and optional legacy-maze runtime |
| [docs/GUIDE.md](docs/GUIDE.md) | Full benchmark matrix, action semantics, protocols and recovery |

Site settings, scheduler adapters, frozen training snapshots and validation tools live outside this repository. The core contains no cluster account settings. Automated checks are a separate package; [VALIDATION.json](VALIDATION.json) records their revision and scope.

## Commands and comparisons

Use `fourier-wm prepare-data|train|eval|submit|doctor|summarize`. `--help` lists options; `--export-config` saves the resolved configuration; `train --dry-run` prints the source training configuration. Nothing launches a full campaign implicitly.

```bash
fourier-wm train --environment tworoom model_family=fourier \
  seed=42 run_name=tworoom_fourier_s42 dataset.manifest=/data/tworoom/train.json
```

Select `model_family=lewm` or `dinowm` for upstream baselines. Fourier ablations use `dynamics_type=linear` or `bilinear` with the same data, representation and training settings. Use separate run names for seeds 42, 43 and 44; compare them on one saved evaluation task list.

Published protocol priority is **SWM > LeWM > DINO-WM**, for the exact benchmark. Push-T and Cube use SWM's Table 1 evaluation; Two-Room uses LeWM's released data and evaluation convention. **Cube requires the LeWM-collected expert release, not the ordinary OGBench play dataset.** Other environments do not inherit results from similarly named benchmarks. The catalog pins source revisions and checksums and rejects unverified substitutions.

Fourier converts position/displacement commands to commanded rates before normalization and blocking. Force/torque commands retain native units and require the validated history path. Baselines keep native actions. Training normalization travels with Fourier weights; it is never refitted on evaluation data.

The existing Two-Room checkpoints and current Two-Room profile supervise one five-step endpoint per clip, plus SIGReg. They are **not** the requested three-endpoint experiment at steps 5, 10 and 15. See the guide before labeling results or starting a new training run.

`planner=cem` evaluates model quality with upstream CEM. `planner=sqp` is the compatibility name for multistart projected Gauss–Newton; `protocol.feedback=true` separately enables TV-LQR. Paper and project closed-loop evaluations have different clipping and batching conventions. Use the [Two-Room commands](docs/TWOROOM.md) or [full guide](docs/GUIDE.md) to select and report them explicitly.
