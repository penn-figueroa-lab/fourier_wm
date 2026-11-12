# Fourier World Model

Fourier dynamics and MPC added around an unchanged, pinned [Stable World Model](https://github.com/galilai-group/stable-worldmodel) checkout. LeWM/DINO-WM reuse upstream models and losses. Fourier replaces LeWM's transformer and prediction head, removes action embedding, and advances five primitive transitions between sparse observations.

## Install

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then:

```bash
bash scripts/install.sh
source .venv/bin/activate
fourier-wm doctor
export FOURIER_DATA=/absolute/path/to/datasets
export FOURIER_RUNS=/absolute/path/to/runs
```

Python 3.11/CUDA 12.8 dependencies are locked. On a cluster, run installation, preparation and computation in allocated jobs. Site settings and scheduler adapters stay outside this repository.

All three model families use a project HDF5 reader that loads only the required image frames. Primitive actions and conversion proprioception stay dense; clip selection, preprocessing and losses stay upstream's. No dataset conversion or extra loader setting is needed.

## Download → train → evaluate

This example uses the released Two-Room data. `TASK` can also be `pusht`, `cube` or `pointmaze`:

```bash
TASK=tworoom
MODEL=fourier
RUN="${TASK}_${MODEL}_s42"

fourier-wm prepare-data --environment "$TASK" --download \
  --output "$FOURIER_DATA/$TASK"

fourier-wm train --environment "$TASK" model_family="$MODEL" \
  seed=42 run_name="$RUN" dataset.manifest="$FOURIER_DATA/$TASK/train.json"

fourier-wm eval --environment "$TASK" seed=42 run_name="${RUN}_cem" \
  checkpoint="$FOURIER_RUNS/$RUN/weights.pt" \
  dataset.manifest="$FOURIER_DATA/$TASK/train.json"
```

**For PointMaze evaluation, use `validation.json` instead of `train.json`.** Its official navigate release has 1,000 training and 100 validation episodes. Downloading preserves both original NPZs; preparation renders their recorded states without stepping physics. Use `--resume-data` to continue an existing image export. `--episodes 1` prepares a limited smoke export; full-data training rejects it.

For upstream baselines, select `MODEL=lewm` or `dinowm`; for Fourier ablations, add `dynamics_type=linear` or `bilinear`. Use separate run names for training seeds 42, 43 and 44. Keep evaluation seed 42 and share the first evaluation's `tasks.json` across models/seeds using `tasks=/absolute/path/to/tasks.json`.

For **Scene**, use `prepare-data --environment scene --download --output "$FOURIER_DATA/scene"`.
This pins the released SWM Lance files (37.65 GB), preserves the originals and makes metadata-only
90/5/5 episode views with split seed 42. Train with `--environment scene`
and both `dataset.manifest="$FOURIER_DATA/scene/train.json"`
and `dataset.validation_manifest="$FOURIER_DATA/scene/validation.json"`;
evaluate with `dataset.manifest="$FOURIER_DATA/scene/test.json"`.
All three families use image observations and native action channels (converted to rates only for Fourier).
Task labels and privileged reset fields never enter the model. The original `raw/` directory must remain
beside the views: they reference its image fragments. Scene is a project benchmark with native all-object
success (cube, buttons, drawer and window), not a Table 1 reproduction or the OGBench RL leaderboard protocol.

Downloads verify pinned SHA-256 hashes, resume partial HTTP transfers and publish completed files atomically. `progress.json` records download/render ETA. Existing unverified files are rejected. To import an existing author release, use `prepare-data --source /data/file.h5 --archive /data/original.tar.zst --output /data/train.json --split train`. Custom tasks use `--collect --source /data/train.h5 --output /data/train.json`; these are project benchmarks.

## Environment settings

There is **one** config directory: [`src/fourier/configs/`](src/fourier/configs). Its [`env/`](src/fourier/configs/env) files contain each environment's physics, action semantics, rendering and dataset provenance. `benchmarks.yaml` contains shared paper profiles, planners and evaluation protocols; `experiment.yaml` contains run settings. `model/` and `train/` adapt the original SWM model/training configs. Commands discover these files independently of the working directory.

| Environment | Data/protocol authority | Training context: LeWM / Fourier | Native command → Fourier input |
|---|---|---|---|
| [Push-T](src/fourier/configs/env/pusht.yaml) | SWM Table 1; author expert release | 3 / 1 image | Relative XY → pixel/s |
| [Cube](src/fourier/configs/env/cube.yaml) | SWM Table 1; **LeWM-collected expert release**, not OGBench play | 3 / 1 image | XYZ, yaw, gripper displacement → separate rates |
| [Two-Room](src/fourier/configs/env/tworoom.yaml) | LeWM; author heuristic release | 1 / 1 image | Relative XY ×5 per 0.1s → pixel/s |
| [OGBench PointMaze](src/fourier/configs/env/pointmaze.yaml) | Official navigate release; project comparison | 3 / 1 image | Displacement ×0.2 per 0.1s → rate |
| [Rope](src/fourier/configs/env/rope.yaml), [Granular](src/fourier/configs/env/granular.yaml) | DINO-WM release; project baseline adaptation | 1 / dense-motion replay required | Complete push endpoints; **not primitive velocity** |
| [Scene](src/fourier/configs/env/scene.yaml) | SWM released visual data; project episode-holdout comparison | 3 / 1 image | XYZ, yaw, gripper displacement → separate rates |
| Fetch, SimplePointMaze | Explicit seeded project collections | 3 / 1 image | Native displacement → configured rates |

The priority is **SWM > LeWM > DINO-WM** for the exact published benchmark. Push-T/Cube paper CEM uses 50 tasks, goal offset 25, budget 50, horizon/replanning 25, and 300 candidates / 30 elites / 30 iterations. Two-Room uses 10 CEM iterations. LeWM-authority training uses 10 epochs; project defaults also use 10. DINO-WM retains its own backbone, action encoder and source optimizer. Paper-labeled DINO-WM retraining with unresolved source/paper conflicts fails explicitly; an alternative requires a project label.

LeWM encodes frames separately before its temporal predictor. Fourier rate-input dynamics use the current frame's latent. Force/torque inputs retain native units and require causal adjacent image history; the optional DINO force-maze path is experimental and needs `scripts/install-legacy-maze.sh`. It is separate from OGBench PointMaze and is not a current campaign target.

Use `fourier-wm train ... --dry-run` to inspect the exact resolved source recipe before training. `--train-option key=value` overrides source settings; paper profiles reject incompatible overrides. `--export-config /path/experiment.yaml` saves the resolved experiment. Layout, clipping, normalization or task changes must be reported rather than silently called the same benchmark.

## Rope and Granular imports

Both tasks share one pinned split ZIP release (about 16 GB). Install the `zip` command, then prepare either or both:

```bash
fourier-wm prepare-data --environment rope --download --output "$FOURIER_DATA/deformable"
fourier-wm prepare-data --environment granular --download --output "$FOURIER_DATA/deformable"
```

The second command reuses the verified download. Existing parts can be supplied with `--source /data/raw_parts` instead of `--download`. Each task produces `rope/` or `granular/` with `train.h5`, `validation.h5`, manifests and conversion progress. Add `--resume-data` after an interrupted conversion. Leave roughly 50 GB free for original parts, the joined archive, source tensors and both converted tasks. Only declared tensor members are read; the archive is never unpacked wholesale.

Imports preserve the released 1,000 × 20 observation/action records per task, integer RGB values, native four-coordinate push actions, particle states and dummy proprioception. The source `torch.randperm(seed=42)` split holds out 100 complete episodes; there is no separate released test split. Particle states are reset/metric data, excluded from model observations. The unchanged SWM reader loads the HDF5s directly. LeWM/DINO-WM training adapters use one-image context, frameskip one, external validation and a project label; this does not claim exact paper reproduction.

**One transition executes an entire variable-duration push** `[start_x,start_z,end_x,end_z]`. Fourier training/SQP/TV-LQR reject these endpoint actions. Dense images, commanded tool motion and timing must be replayed and validated before training our primitive-rate dynamics. Subtracting endpoints loses the push's contact location and is insufficient.

Optional baseline CEM execution uses the DINO-WM PyFlex backend at commit `0a9492fa12044b852ae9e001cc74604b79c8bb0c`, selected by `DINO_WM_ROOT`. Follow that project's optional deformable-runtime installation separately; data import needs no PyFlex, Gym or PyBullet installation. The bridge requires one environment per process. Real PyFlex replay and B200/MIG compatibility must pass on allocated hardware before launching evaluations. The default is a **project** held-out dataset-goal protocol with five complete pushes, replanning after each; Chamfer thresholds come from LaLa-World, whose task sampler and published rates are not interchangeable with this protocol. `doctor --environment rope` reports these boundaries.

## Test shared weights

Keep `weights.pt`, `config.json` and `training_metadata.json` together:

```bash
fourier-wm doctor --environment tworoom checkpoint=/shared/tworoom_fourier_s42/weights.pt
python scripts/demo_tworoom.py --checkpoint /shared/tworoom_fourier_s42/weights.pt \
  --output "$FOURIER_RUNS/tworoom_demo"
```

The dataset-free demo saves a GIF with current observation left and goal right. Add `--planner sqp --feedback` and choose a new output directory to try eight-start Gauss–Newton with TV-LQR. It is one native seeded task, not a paper success-rate evaluation.

For exact saved-task replay, copy the handoff's original `dataset/train.json`, change **only its `path`** to your local HDF5, and pass the handoff's `evaluation/tasks.json` to `eval`. Schema-v2 identity excludes the file locator; data hashes remain verified. Recreating a manifest may change its provenance identity. Fourier always uses saved training normalization, never statistics refitted on test data.

**Existing Two-Room weights:** each clip supervises `E(o_t) → five Fourier transitions → E(o_(t+5))`, with endpoint MSE + 0.09 SIGReg. Gradients flow through all five steps and the encoder. These weights/current Two-Room profile use one pair per clip, not the requested three pairs through step 15. Push-T/Cube use three teacher-forced five-step pairs; this is not an uninterrupted fifteen-step rollout. Do not relabel existing experiments.

## Closed-loop control

```bash
fourier-wm eval --environment tworoom protocol=closed_loop planner=sqp \
  seed=42 run_name=tworoom_gn_tvlqr \
  checkpoint=/shared/tworoom_fourier_s42/weights.pt dataset.manifest=/data/tworoom/train.json \
  tasks=/shared/tworoom_fourier_s42/evaluation/tasks.json \
  protocol.num_eval=50 planner.starts=8 protocol.replan_steps=5 protocol.feedback=true \
  protocol.horizon_schedule=deadline
```

`sqp` is the compatibility name for projected Gauss–Newton. TV-LQR is a separate correction at every observation. `deadline` uses horizons 25,20,15,10,5 between 25-step deadlines. For a matched project CEM comparison, use `planner=cem`, omit `planner.starts`, and disable feedback. Paper CEM preserves upstream execution; project MPC clips executed actions, and Gauss–Newton also projects its search. Report the distinction.

For longer PointMaze tasks, select `protocol=closed_loop`, set `protocol.goal_offset=50 protocol.horizon_steps=50 protocol.budget=100`, and use a shared validation task list. For 75 steps, use offset/horizon 75 and budget 150. History-state Fourier checkpoints currently support CEM only; single-state SQP/TV-LQR cannot substitute.

## Outputs and recovery

Each run has immutable config/provenance, dataset and normalization identities, metrics, checkpoints and evaluation records. Output directories cannot be reused accidentally. Send `SIGUSR1` to checkpoint and exit; resume the same source/config with `resume_from=/runs/name/resume/last.ckpt`. Code/config mismatches are rejected. Existing PARCC snapshots stay frozen for exact resume; a cleaned checkout is not silently substituted.

`fourier-wm summarize` reads saved results. Videos are optional with `output.save_video=true`. `submit` uses an externally installed site adapter. Tests are a separately versioned package; [VALIDATION.json](VALIDATION.json) records checks and remaining limits. Full benchmark success is not established by smoke tests.

Project code lives only in `src/fourier/`; `scripts/` contains installation and the small weight demo. `stable-worldmodel/` retains the exact upstream tree and licensing from `UPSTREAM.json`. Data, weights, generated results and cluster operations stay outside Git.
