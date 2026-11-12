# Fourier World Model

Fourier dynamics and MPC around an unchanged, pinned [Stable World Model](https://github.com/galilai-group/stable-worldmodel). LeWM and DINO-WM reuse upstream architectures and losses. Fourier replaces LeWM’s transformer and prediction head, removes action embedding and makes five primitive transitions between sparse observations.

## Layout

| Path | Contents |
|---|---|
| `src/fourier/` | Dynamics, action conversion, training, planners and evaluation |
| `src/fourier/configs/` | **The only config directory**: environment files, model/training adapters and shared benchmark profiles |
| `scripts/` | Installation and a Two-Room checkpoint demo |
| `stable-worldmodel/` | Unmodified upstream source; revision and license recorded in `UPSTREAM.json` |

Datasets, weights, results, cluster settings and validation tools stay outside this repository. Work on `main`; completed experiments retain their original immutable source snapshots for exact resume.

## Install

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then:

```bash
bash scripts/install.sh
source .venv/bin/activate
fourier-wm doctor
export FOURIER_DATA=/absolute/path/to/datasets
export FOURIER_RUNS=/absolute/path/to/runs
```

Python 3.11/CUDA 12.8 dependencies are locked. On clusters, install and compute in allocated jobs. Commands find packaged configs independently of the working directory. All model families use a reader that loads only required image frames while preserving dense actions and conversion proprioception.

## Download → train → evaluate

For `tworoom`, `pusht`, `cube` or `pointmaze`:

```bash
TASK=tworoom
MODEL=fourier  # Also lewm or dinowm
RUN="${TASK}_${MODEL}_s42"
fourier-wm prepare-data --environment "$TASK" --download --output "$FOURIER_DATA/$TASK"
fourier-wm train --environment "$TASK" model_family="$MODEL" seed=42 \
  run_name="$RUN" dataset.manifest="$FOURIER_DATA/$TASK/train.json"
fourier-wm eval --environment "$TASK" seed=42 run_name="${RUN}_cem" \
  checkpoint="$FOURIER_RUNS/$RUN/weights.pt" dataset.manifest="$FOURIER_DATA/$TASK/train.json"
```

**PointMaze evaluates on `validation.json`.** Its official navigate release has 1,000 training and 100 validation episodes; preparation renders recorded states without stepping physics. Keep the original NPZs. `--resume-data` resumes an interrupted export; `--episodes 1` makes a smoke export that full training rejects.

Train seeds 42, 43 and 44 with separate run names. Evaluation uses seed 42 and the same saved `tasks.json` for all models/seeds: pass `tasks=/absolute/path/to/tasks.json`. Fourier linear/bilinear ablations add `dynamics_type=linear` or `bilinear`.

Use `train ... --dry-run` to inspect the resolved recipe, `--export-config /path/experiment.yaml` to save it and `--train-option key=value` for explicit source overrides. Paper profiles reject incompatible changes; label alternative protocols as project benchmarks.

Downloads resume partial transfers, verify pinned SHA-256 hashes and publish atomically. Progress JSONs record ETA. Existing unverified files are rejected. Existing author files can be imported with `prepare-data --source /data/file.h5 --archive /data/original.tar.zst --output /data/train.json --split train`; seeded custom collections use `--collect --source /data/train.h5 --output /data/train.json`.

## Benchmarks and controls

The priority is **SWM > LeWM > DINO-WM** for an exact published setup. Each [`env/`](src/fourier/configs/env) file records physics, action units, rendering and provenance; [`benchmarks.yaml`](src/fourier/configs/benchmarks.yaml) records shared training/evaluation profiles.

| Environment | Protocol authority | Images: LeWM / Fourier | Native command → Fourier input |
|---|---|---|---|
| Push-T | SWM Table 1; expert release | 3 / 1 | Relative XY → pixel/s |
| Cube | SWM Table 1; **LeWM-collected expert release**, not OGBench play | 3 / 1 | XYZ, yaw, gripper displacement → separate rates |
| Two-Room | LeWM heuristic release | 1 / 1 | Relative XY ×5 per 0.1s → pixel/s |
| OGBench PointMaze | Official navigate release; project comparison | 3 / 1 | Displacement ×0.2 per 0.1s → rate |
| Scene | SWM visual release; project episode holdout | 3 / 1 | XYZ, yaw, gripper displacement → separate rates |
| Rope / Granular | DINO-WM release; project baseline adaptation | 1 / not ready | Complete push endpoints; Fourier needs dense-motion replay |
| Fetch / SimplePointMaze | Explicit seeded project collections | 3 / 1 | Native displacement → configured rates |

Push-T/Cube paper CEM uses 50 tasks, goal offset 25, budget 50, horizon/replanning 25 and 300 candidates / 30 elites / 30 iterations. Two-Room uses 10 CEM iterations. These settings are not defaults for unrelated tasks. LeWM-authority and project training budgets are 10 epochs; DINO-WM retains its backbone, action encoder and source optimizer. Unresolved paper/source conflicts reject paper-labeled DINO-WM retraining.

LeWM encodes history images separately. Fourier rate-input dynamics use the current image. Native force/torque inputs require causal adjacent image latents, not block-spaced history or conversion to velocity. The experimental DINO force-maze path (`scripts/install-legacy-maze.sh`) is separate from OGBench and outside the current campaign.

**Existing weights:** Two-Room clips supervise one pair `E(o_t) → five Fourier transitions → E(o_(t+5))`, with MSE + 0.09 SIGReg. Gradients flow through the encoder and all five transitions. Push-T/Cube supervise three teacher-forced five-step pairs through observation 15; this is not one uninterrupted 15-step rollout. Preserve these distinctions when sharing results.

## Scene, Rope and Granular

All use the same CLI; train with both `dataset.manifest=.../train.json` and `dataset.validation_manifest=.../validation.json`.

| Task | Preparation output | Training / evaluation split | Readiness |
|---|---|---|---|
| Scene | `prepare-data --environment scene --download --output "$FOURIER_DATA/scene"` | 90/5/5% episodes; evaluate `test.json` | Three model families; native all-object success |
| Rope | `prepare-data --environment rope --download --output "$FOURIER_DATA/deformable"` | 900/100 episodes; evaluate `rope/validation.json` | LeWM/DINO-WM offline training; PyFlex evaluation needs runtime validation |
| Granular | Same command with `--environment granular` | 900/100 episodes; evaluate `granular/validation.json` | Same limits as Rope |

Scene pins the 37.65 GB SWM Lance release. Keep its original `raw/` directory beside metadata-only split views. All models receive images; privileged reset/task fields do not enter the model. This is a project benchmark, not Table 1 or the OGBench RL leaderboard.

Rope/Granular share one 16 GB split ZIP; install `zip` and leave roughly 50 GB free for both imports. The second command reuses the download. `--source /data/raw_parts` imports existing parts, and `--resume-data` continues conversion. The importer selectively reads source tensors and preserves RGB, native controls and particle states. It uses the source seed-42 episode split; no separate released test split exists. Particle states are reset/metric data only.

For example, train a Rope baseline with:

```bash
fourier-wm train --environment rope model_family=lewm seed=42 run_name=rope_lewm_s42 \
  dataset.manifest="$FOURIER_DATA/deformable/rope/train.json" \
  dataset.validation_manifest="$FOURIER_DATA/deformable/rope/validation.json"
```

**A deformable action executes a complete variable-duration push** `[start_x,start_z,end_x,end_z]`. Fourier training/SQP/TV-LQR reject these controls. Dense commanded tool motion, timing and intermediate observations must be replayed and validated first; subtracting endpoints loses contact location. Do not queue Fourier on this endpoint release.

Optional baseline CEM uses DINO-WM PyFlex commit `0a9492fa12044b852ae9e001cc74604b79c8bb0c`, selected through `DINO_WM_ROOT` and installed separately. One environment per process is supported. Native replay and GPU compatibility must pass before evaluation. The project protocol uses five complete pushes, replanning after each, and LaLa-World Chamfer thresholds; its task sampler and results are not interchangeable with published success rates. `doctor --environment rope` reports these limits.

## Share weights and run MPC

Keep `weights.pt`, `config.json` and `training_metadata.json` together:

```bash
fourier-wm doctor --environment tworoom checkpoint=/shared/tworoom_fourier_s42/weights.pt
python scripts/demo_tworoom.py --checkpoint /shared/tworoom_fourier_s42/weights.pt \
  --output "$FOURIER_RUNS/tworoom_demo"
```

The dataset-free demo saves a current-observation/goal GIF. Add `--planner sqp --feedback` for eight-start Gauss–Newton with TV-LQR; one demo is not a paper evaluation. For exact replay, copy the original manifest, change only its `path`, and reuse the original task list. Fourier loads saved training normalization and never refits it on evaluation data.

```bash
fourier-wm eval --environment tworoom protocol=closed_loop planner=sqp \
  seed=42 run_name=tworoom_gn_tvlqr checkpoint=/shared/tworoom_fourier_s42/weights.pt \
  dataset.manifest=/data/tworoom/train.json tasks=/shared/evaluation/tasks.json \
  protocol.num_eval=50 planner.starts=8 protocol.replan_steps=5 protocol.feedback=true \
  protocol.horizon_schedule=deadline
```

`sqp` names projected Gauss–Newton for compatibility. TV-LQR corrects each observation separately. `deadline` shrinks horizons 25,20,15,10,5 between deadlines. Project CEM uses `planner=cem`, no `planner.starts` and feedback disabled. Project execution clips actions; Gauss–Newton also projects its search. Paper CEM preserves upstream behavior.

Long PointMaze tasks use `protocol=closed_loop protocol.goal_offset=50 protocol.horizon_steps=50 protocol.budget=100` with shared validation tasks. Use 75/75/150 for longer goals. History-state Fourier checkpoints support CEM only; single-state SQP/TV-LQR cannot substitute.

## Outputs and recovery

Runs save immutable config/provenance, data/normalization identities, metrics, weights, recovery checkpoints and evaluation records. Names cannot collide. `SIGUSR1` saves and exits; resume with `resume_from=/runs/name/resume/last.ckpt` under the original source/config. Incompatible resumes are rejected.

`fourier-wm summarize` reads saved results. Videos are optional (`output.save_video=true`); `submit` uses an external site adapter. [VALIDATION.json](VALIDATION.json) records the separately versioned checks and remaining limits. Passing pipeline checks does not establish benchmark success.
