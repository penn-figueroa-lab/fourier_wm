# Models, commands and evaluation protocols

`stable-worldmodel/` is the unmodified tree at upstream commit `21446f1ede6d5284e981bd7b47f432b994e6d812`. `UPSTREAM.json` records provenance; `fourier-wm doctor` reconstructs the Git tree and checks all 357 source files. The three integration edits that followed the original import have been removed from that directory. Upstream licensing remains in place. Extensions live under `src/fourier/`; `configs/` points to the packaged configurations.

## Baselines

LeWM imports the original encoder, transformer predictor, projector, `lejepa_forward`, SIGReg and preprocessing. DINO-WM uses the original frozen DINO backbone, PreJEPA predictor, `dinowm_forward` and target detachment. Native dataset actions remain unchanged for both. Optimizer, image size, training history, source clip split and loss defaults come from the pinned source configurations. Explicit `--train-option` overrides are recorded; reduced smoke configurations are not published baseline results.

The LeWM paper specifies 10 epochs, batch 128, ViT-tiny, 224px and action blocks of five. Its training history is three for Push-T/Cube and one for TwoRoom; the LeWM-authority DINO-WM comparison excludes proprioception. The shared adapters retain source models/losses and apply these explicit environment overrides. The generic source LeWM default is 100 epochs. The source DINO-WM adapter defaults to 10 epochs, but the original DINO-WM paper and training config specify 100; those settings are not interchangeable.

An encoder processing one image per call is different from a predictor conditioning on several encoded images. LeWM encodes frames independently, then its causal transformer uses up to three frame embeddings on Push-T/Cube, spaced five primitive steps apart ([Appendix D](https://arxiv.org/html/2603.19312v3#A4)); TwoRoom retains the published one-frame setting. The rate-input Fourier predictor advances each current image latent independently; having three supervised forecasts in a training clip does not give that predictor image history. For force/torque inputs, the Fourier state instead concatenates image latents. The force-maze preset uses adjacent physical frames [z_t,z_(t-1),z_(t-2)], with one primitive step between frames. The history shifts exactly after every predicted primitive transition. The same spacing, state construction and loss apply to linear/bilinear ablations. Training context and evaluation context are recorded separately: the pinned SWM paper evaluator defaults to one context frame.

`paper_priority` and `published` in the benchmark catalog enforce **SWM > LeWM > DINO-WM** for published control results. Being listed in SWM's environment registry does not establish a published control benchmark. SWM Appendix I delegates training to original publications: Fourier/LeWM therefore use LeWM's 10-epoch budget. Cube evaluation uses the pinned released SWM config because Appendix I details only Push-T. Paper profiles check source data, frame spacing, history, full training budget, batch size, encoder and key optimizer/loss settings; changed settings require an explicit project label.

Known unresolved source conflicts remain visible in that catalog. SWM's generic DINO adapter uses 10 epochs whereas its stated original-publication delegation points to 100; paper-labeled DINO retraining for SWM is blocked pending resolution. Original DINO-WM also has paper/code differences in predictor learning rate (5e-5 versus 5e-4) and conflicting image-size statements. Its native environment/data/evaluation adapters are not yet validated here, so those benchmark names fail rather than silently selecting project mazes. The original DINO loader splits episodes; SWM's loader splits clips. Training/evaluation settings must come from one declared benchmark.

Fourier replaces only the predictor and prediction-output projection. For Push-T/Cube, the three forecasts from `o_t,o_{t+5},o_{t+10}` target `o_{t+5},o_{t+10},o_{t+15}`. The TwoRoom paper profile uses history one and a single five-step forecast. Each forecast recursively applies five primitive transitions. These are three teacher-forced five-step forecasts, not one uninterrupted fifteen-step rollout. The loss is upstream endpoint MSE plus 0.09 SIGReg, with gradients to the encoder/projector and Fourier dynamics. No decoder or generative objective is added by this release.

Baseline evaluation retains upstream evaluation-dataset normalization. Fourier uses saved training normalization and never refits it on evaluation data. Reports identify this difference. External validation manifests use training transforms; they cannot reuse the training file under a new split label.

## Command coordinates

For each primitive action, before normalization and blocking:

- Native velocity: `u = scale * a` (scale 1 preserves native units).
- Relative displacement: `u = scale * a / dt`.
- Absolute target: `u = (scale * a - q_current) / dt`.
- Force or torque: `u = scale * a`, retaining native command semantics; no division by `dt`. Motor commands are not automatically forces in newtons.

`ActionAdapter` supplies the inverse execution mapping and differentiable Torch conversion. Translation, yaw and gripper channels have separate units/scales/bounds in each environment config. Absolute targets require dense current-step proprioception and explicit channel indices. Future observed displacement is never an action label. Raw files are preserved; dense conversions are lazy derived views whose recipe and identity are recorded in the manifest.

The native simulator controller, contact dynamics and saturation remain active. Commanded rate is not achieved velocity. An absolute-target adapter needs live proprioception on each execution step even if MPC replans less often; this additional feedback must be reported. Local analytic derivatives do not differentiate through contact or native clipping. Native clipping counts are reported. Discrete tasks are excluded. Force/torque coordinate conversion is tested, but the shared Fourier training entrypoint rejects inertial tasks except the DINO-WM force-maze path described below, which uses causal image history for LeWM/Fourier. The original released DINO-WM Maze model separately encodes measured position/velocity. History helps infer hidden motion; a pair of frames gives average latent motion, not guaranteed instantaneous physical velocity. Position/velocity commands alone do not guarantee that a single image makes a task fully observable.

## Presets

| Preset | Meaning |
|---|---|
| `swm_table1` | Push-T and Cube: 50 tasks, 25/50 goal offset/budget, horizon/replanning 25, CEM 300 candidates / 30 elites / 30 iterations. Cube settings follow the pinned SWM evaluator; the paper's Appendix I explicitly details Push-T |
| `swm_paper` | Push-T only: 50 trajectories, goal offset 25, budget 50, horizon 25 physical steps in five blocks, replanning every 25, CEM 300 candidates / 30 elites / 30 iterations |
| `lewm_paper` | Author data: 25/50 goal offset/budget for Push-T, Cube and TwoRoom. Push-T uses 30 CEM iterations; Cube/TwoRoom use 10. Each uses 300 candidates, 30 elites and a 25-step horizon |
| `dataset_goal` | Project comparison on saved starts/goals; choose and record each environment's offset, budget and horizon |
| `closed_loop` | Project comparison with explicit primitive-step replanning and optional Fourier TV-LQR |

The last two presets contain project defaults. Environment selection defaults to `swm_table1` for Push-T/Cube, `lewm_paper` for TwoRoom and `dataset_goal` for other families. `swm_paper` remains a Push-T compatibility preset. Paper evaluation uses history_len=1, matching the pinned source evaluator; training follows the environment-specific history above. Layouts, Cube variants and all changes are included in dataset identity and must match evaluation. See the [published SWM experimental setup](https://arxiv.org/html/2605.21800v1#A9).

For Table 1, expert training-release start/goal tasks and held-out expert-validation tasks are distinct regimes. Do not rename the former a held-out test. Internal random clip validation follows the source code and can share episodes with fitting. For a project environment, collect separate train/validation/test episodes and give every model the same manifests and saved test tasks. PointMaze now defaults to the official OGBench `pointmaze-large-navigate-v0` release. Training rejects the earlier waypoint collection under that default. Synthetic checks can select `dataset.regime=project_custom`, random collection and a seed explicitly. The superseded waypoint collector is removed; it cannot be selected to regenerate the retired data.

The common `seed` reaches every model's initialization, split and loader. Use three independent seeds as an initial variance estimate, keep evaluation tasks and planner seed fixed, and report per-seed values plus mean/sample standard deviation. Published baseline numbers may be cited without rerunning them, but cannot be described as our own matched-seed replication. `machine.num_workers=null` preserves source defaults; explicit machine settings or `--train-option` settings are recorded. `health.json` records finite training metrics, step, epoch, learning rate and GPU peak memory once per minute without adding an objective or changing gradients.

The pinned Cube HDF5 stores 10,000 episodes with 201 rows each: 200 transitions plus a terminal observation whose action is NaN. Thus the release has 2,010,000 stored frames. Catalog checks distinguish these stored-row counts from the paper’s transition count; the released records are preserved.

Project benchmarks default to 10 epochs for every model; an explicit `trainer.max_epochs` training override selects a different budget. `training_profile=source` retains the upstream budget. Model-specific batch sizes and optimizers remain native, so equal epochs mean matched dataset passes rather than equal optimizer-update counts.

OGBench PointMaze uses only the official `pointmaze-large-navigate-v0` release: 1,000 training and 100 released validation episodes, each with 1,001 stored observations and 1,000 usable transitions. The earlier waypoint collection, its checkpoints/results and its collection/expansion tools have been retired and deleted. No waypoint score is part of the current benchmark. The native `ob_type=pixels` rendering omits the task-goal marker, and `observation_keys: [pixels]` restricts all models to images and recorded actions. DINO-WM's extra encoder contains actions only. All three models use saved training action statistics during evaluation; coordinate-conditioned checkpoints are rejected. The released validation partition remains separate from training; saved evaluation tasks must be finalized and recorded before the new campaign starts.

Exact task records include dataset hash, episode, starting frame, goal offset and reset seed. Share `tasks.json` between models and planners; count/range/identity mismatches fail. State is used only to reset the simulator and evaluate its native success criterion. Planner observations are explicitly restricted to images, required proprioception and executed-action history. The evaluator reports initially solved cases separately, preserves native success tests, and caps execution by primitive steps.

Paper presets delegate to the unchanged upstream batched evaluator and original baseline policy. Project dataset-goal/closed-loop evaluation executes tasks sequentially; this changes CEM random-number consumption across tasks. Reports identify batch size; do not mix those protocols. Do not compare a source result to a project result as if their random streams and normalization were identical. Published baseline settings are preserved, but benchmark reproduction requires the actual released evaluation data and matching distribution-shift settings.

## Controller

The historical `sqp` name denotes **multistart projected Gauss–Newton**, not a general constrained SQP implementation. It minimizes terminal latent goal error plus configured quadratic action-difference/magnitude penalties. At each iteration it forms the rollout action Jacobian, solves a damped Gauss–Newton system, evaluates a fixed backtracking set, projects into normalized action bounds and retains the lowest-cost candidate. Starts include shifted previous commands, zeros and seeded random starts. It uses the exact Fourier transition and analytic action derivatives in JAX; external checks compare Torch/autograd/JAX.

MPC encodes the latest observation, optimizes a finite primitive-action sequence, executes `replan_steps`, shifts the remaining warm start and repeats. Context uses actual observations at block spacing and the actions actually executed. Resets clear only the affected environment. Fourier+CEM isolates model quality from the specialized controller. DINO's goal cost excludes action slots and separately averages pixel/proprioception errors.

Optional TV-LQR builds a nominal latent rollout, computes local state/action Jacobians and solves a finite-horizon Riccati recursion. On every observation it applies `u_nominal - K (z_observed - z_nominal)` before conversion/clipping. This is feedback, not a safety certificate. It is configured separately from MPC replanning.

Timing records distinguish JAX compilation, planning excluding compilation, feedback, total policy time, and rendering/observation-to-action latency. The first call still includes noncompiler cold-start overhead; compare warmed calls separately. No speed or success advantage is claimed by the short pipeline tests.

Exact CUDA recovery tests explicitly enable Lightning deterministic execution and `CUBLAS_WORKSPACE_CONFIG=:4096:8`. Default CUDA kernels can differ numerically across restarts even with identical RNG and sample order; use the recorded deterministic setting when bitwise recovery is required. This is not silently imposed on published baseline training.

Released upstream baseline checkpoints need only their original `config.json` and weights. Their native action-block width is read from that configuration. Current Fourier checkpoints additionally require training normalization metadata; older `FourierWorldModel` checkpoints can use their embedded mean/scale buffers. Missing Fourier training statistics are an error, never an invitation to fit evaluation data. Reports distinguish native/legacy coordinates from the new commanded-rate interface.

## Dataset catalog and supported paths

`configs/benchmarks.yaml` pins each author archive, revision, checksum, expected episode/frame counts and environment action recipe. Download the named artifact from that revision before preparation; there is no silent download or substitution. Cube contains 10,000 expert episodes of 200 steps. The released Push-T file contains 18,685 episodes and 2,336,736 frames, despite the paper's description of 20,000 episodes. This release discrepancy is recorded explicitly. TwoRoom contains 10,000 heuristic episodes. Sources: [LeWM Appendix E/F](https://arxiv.org/html/2603.19312v1#A5), [author Cube release](https://huggingface.co/datasets/quentinll/lewm-cube), [Push-T release](https://huggingface.co/datasets/quentinll/lewm-pusht), [TwoRoom release](https://huggingface.co/datasets/quentinll/lewm-tworooms).

Preparation checks the extracted HDF5 bytes against the approved unpacked hash or streams the checksum-verified source archive to establish the correspondence. Preserve raw files and archives until verification finishes. The verified manifest records episode boundaries, preprocessing, action ranges, environment/dependency versions and split policy. Some author commands exceed advertised Gym action bounds: training preserves them and records the count. It does not clip or change the released dataset.

| Environment selector | Native command → Fourier input | Data status |
|---|---|---|
| `pusht` | Relative XY ×100 per 0.1s → pixel/s | Pinned author expert release |
| `cube` | Relative XYZ/yaw/gripper per 0.05s → m/s, rad/s, opening/s | Pinned single-cube author expert release |
| `tworoom` | Relative XY ×5 per 0.1s → pixel/s | Pinned author heuristic release |
| `pointmaze` | OGBench point displacement ×0.2 per 0.1s → rate | Official navigate NPZ → native goal-free image export; layout explicit |
| `simple_pointmaze` | Separate SWM point displacement → units/step | Seeded project benchmark |
| `scene` | XYZ/yaw/gripper → separate rates | Project collection/export |
| `fetch_reach`, `fetch_push`, `fetch_slide`, `fetch_pickandplace` | XYZ ×0.05 and gripper per 0.04s → separate rates | Project collection/export |

All ten paths have external headless reset/replay and short train/planning checks. These do not establish benchmark performance on untested releases. Other force/torque training remains gated as described above; discrete tasks are excluded. The OGBench maze release is distinct from the LeWM Cube visual expert release. A state-only Cube NPZ cannot substitute for that author benchmark. Layouts/manipulation variants must match the manifest exactly.

**The PointMaze benchmarks are different.** This project's `pointmaze` is OGBench's displacement-controlled point on an explicit maze layout; `simple_pointmaze` is a separate SWM kinematic environment. DINO-WM's [published Maze benchmark](https://arxiv.org/html/2411.04983v2#A1) uses a force-actuated ball with inertia, 2,000 random trajectories of 100 steps, and random-state evaluation goals. The authors [release its data and checkpoints](https://huggingface.co/datasets/gaoyuezhou/dino_wm). This is newly collected random data in a D4RL-derived environment, rather than the original D4RL offline training dataset. Its reported 98% MPC success is a result for that benchmark, not a baseline score for either project environment. LeWM's [latest paper](https://arxiv.org/html/2603.19312v3) reports Push-T, OGBench-Cube, Two-Room and Reacher, with no PointMaze benchmark. Reusing the DINO-WM number requires its native environment, data and evaluation protocol, plus a separately validated force-input extension; relabeling force as velocity is invalid. Project PointMaze comparisons therefore retain matched baseline training.

The portable importer writes SWM-compatible HDF5 and provenance manifests from the two checksum-pinned original maze NPZs. It preserves 1,000 training episodes and 100 validation episodes, each containing 1,001 observations and 1,000 usable transitions. Rendering restores recorded states without stepping physics; terminal actions become NaN only in the derived export. The original files remain unchanged. Run rendering on a compute node when using a cluster:

```bash
fourier-wm prepare-data --environment pointmaze --catalog ogbench_pointmaze_navigate \
  --source /data/raw/ogbench --output /data/visual/ogbench_pointmaze_large_navigate_v0
# After interruption, repeat the same command with --resume-data.
```

The importer uses atomic publication, an exclusive output lock, checksum checks and a flushed rendering cursor. `--episodes 1` is a limited smoke export and is rejected by default full-data training. Official OGBench navigation tasks and the DINO-WM force U-maze use separate data identities; neither inherits the other’s published success score. Original DINO-WM’s released Maze loader also provides `(x,y,vx,vy)` proprioception; an image-only adaptation must be labeled separately from that released configuration.

For SWM Table 1, the reported LeWM/DINO-WM scores are 94%/92% on Push-T and 72%/86% on Cube. Expert-train evaluation is the in-distribution regime; expert-validation and random-policy evaluations are separate. Matching data and protocol permits comparison with those published numbers, not a claim to reproduce them exactly on different tasks, seeds or hardware.

Custom data must be explicitly labeled, for example:

```bash
fourier-wm prepare-data --environment simple_pointmaze --collect --episodes 20 \
  --source /data/simple_train.h5 --output /data/simple_train.json --split project_train
```

Separate held-out collections require distinct seeds/files. Baseline default training retains the original random clip split, which can overlap episodes; external validation manifests require distinct files and reuse training normalization. Schema-v2 manifest identity excludes only the local file locator, so data can move to another machine without changing task identity. Content is rehashed on load.

## Comparing controllers and released checkpoints

Paper CEM preserves upstream execution semantics without wrapper clipping, for Fourier as well as baselines; native simulator saturation still applies. Project primitive MPC clips execution to native bounds. Gauss–Newton also projects normalized commands to its configured bound (default 3). Reports identify execution clipping. Compare Fourier+CEM and Fourier+GN with matching project execution policy and report search-bound differences. The legacy Fourier checkpoint uses native-action normalization; it is labeled accordingly rather than relabeled as rate-trained.

Use `fourier.checkpoints.load_pretrained` for released checkpoints from older Transformers versions. The adapter strictly renames ViT parameter keys and rejects missing/unexpected weights; it does not change tensors or architecture. The original upstream loader remains unchanged. Current checkpoint weights/config interfaces are retained.

## External verification

Install the compatible separately versioned validation wheel recorded in `VALIDATION.json`, then run `fourier-wm-validate -q`. Checks cover upstream-tree/baseline parity, all action mappings and environment resets, endpoint gradients, Torch/JAX derivatives, causal MPC histories, DINO action-free scoring, atomic/concurrent output handling, and interrupted/resumed CPU and CUDA training. Site-specific scheduler checks belong to the optional private adapter.

Real trained-checkpoint evaluations are recorded separately from smoke tests. A passing short test does not prove convergence, paper performance, distribution-shift robustness or a controller speed advantage. Hardware-specific readiness requires a completed allocated test on that hardware; it cannot be inferred from the local GPU.

For data without recorded reset seeds, the evaluation adapter supplies each saved task's seed through a Gym reset wrapper. Upstream otherwise resets with `seed=None`; collision state can then differ even for an identical saved start row. This intentional reproducibility fix is recorded as `reset_seed_policy=dataset_seed_or_saved_task_seed`, and is a protocol difference from an unseeded source invocation. The vendor source itself is unchanged. Set `evaluation_batch_size=protocol.num_eval` as an integer (for example `50`) to use the same batched loop for a project controller comparison on Push-T, Cube or TwoRoom; other environment paths currently require batch size 1. Project defaults remain sequential.

Imported data's actual collection description appears in `collection_settings`. `environment.collection` is the configuration template for explicitly collecting new project data; it is not evidence that an imported expert release was collected randomly. Unknown configuration keys fail; use an explicit `+key=value` only when deliberately extending a constructor configuration.


### Original DINO-WM force maze

`environment=dinowm_pointmaze` is a separate benchmark from OGBench `pointmaze`. The catalog pins the original DINO-WM random-force release: 2,000 episodes × 100 frames, split into the original 1,800 training and 200 validation episodes before clip extraction. Conversion preserves original pixels, `(x,y,vx,vy)` proprioception, native forces and episode IDs in SWM HDF5. Prepare the two converted artifacts with `--catalog dinowm_maze`, `--split train` and `--split validation`; their approved checksums are split-specific. An ordinary D4RL offline dataset is not a substitute for this author's collection.

Run the original MuJoCo 2.1 physics in a separate Python 3.9 process. Training/inference remains in the locked Python 3.11 environment:

```bash
FOURIER_LEGACY_ROOT=/data/runtimes/d4rl-maze bash scripts/install-legacy-maze.sh
source /data/runtimes/d4rl-maze/runtime.env
fourier-wm train --environment dinowm_pointmaze model_family=fourier seed=42 \
  run_name=maze_fourier_s42 dataset.manifest=/data/maze/train.json \
  dataset.validation_manifest=/data/maze/validation.json
# Repeat with model_family=lewm and distinct run names/seeds.
```

The installer records exact Python packages and extracted OS-library checksums outside the repository; it requires no root installation. On a cluster, run installation, replay and evaluation in allocated compute jobs.

The LeWM/Fourier comparison consumes images and native forces. This follows the original [LeWM observation encoder](https://github.com/lucas-maes/le-wm/blob/8edfeb336732b5f3ce7b8b210d0ba370a09e2cac/jepa.py): it encodes pixels and embeds actions, without a measured position/velocity encoder. It is not a shared image-only requirement imposed on every baseline. Dataset position/velocity columns remain available for simulator resets and independent diagnostics; they never enter the LeWM/Fourier encoder, predictor, or goal embedding. LeWM uses its unchanged image encoder, observation projector, transformer predictor, action embedding, prediction head and source loss. `ProprioLeWM` is our legacy observation extension, not the original LeWM architecture; it is retained only to load legacy checkpoints.

Fourier shares the LeWM image encoder, observation projector and loss. Its replacement prediction branch accepts raw normalized force blocks without a learned action embedding. `AdjacentHistoryPredictor` forms the 576-dimensional state [E(o_s),E(o_(s-1)),E(o_(s-2))] from 192-dimensional frame latents. This construction stays inside the dynamics predictor and adds no observation adapter or separate MLP. Missing past observations repeat the earliest available image; later observations are never used to infer earlier motion. Each primitive transition predicts only the next current-image latent and shifts the preceding two frame latents exactly. Five transitions connect each teacher-forced source history to its sparse target. The loss supervises endpoints 5,10,15 and adds 0.09 SIGReg on the four endpoint image latents. Training loads adjacent source-history images without changing the sparse target spacing, episode split or clip indexing; no extra intermediate reconstruction or latent losses are added. During planning the full primitive history persists across action-block boundaries. Actual observations update a primitive-step history buffer, independent of the replan interval. Native force inputs are never divided by the control interval. An adjacent-history clip encodes ten source/target images. For batch128, start with two data-loader workers and `loader.prefetch_factor=1`; record these as execution settings. Large worker/prefetch combinations can exhaust host RAM even when GPU memory fits. Older difference-encoded checkpoints retain their original interpretation through the versioned `history_encoding` field; new force training selects the separate adjacent-history classes. Legacy block-spaced concatenation/difference classes retain their checkpoint interpretation.

Linear and bilinear ablations change only `dynamics_type`. They share the Fourier encoder/projector, history encoding and spacing, native command conversion, normalization, original data splits, primitive action blocks, endpoint losses, SIGReg, optimizer, seeds and evaluation tasks. For position/velocity commands all three dynamics variants use one image latent; for the force-maze preset all three use the same concatenated three-latent state.

The source preset uses 100 epochs, training history three, batch 128 and the LeWM optimizer. Evaluation uses original random-state task sampling (seed 99), three-frame causal image context (primitive stride1 for Fourier/linear/bilinear; source stride5 for LeWM), CEM 300/30/10 and 25-step execution blocks. History-aware Fourier evaluation currently uses CEM; the existing single-state analytic SQP/TV-LQR path fails explicitly rather than using inconsistent dynamics. The current 250-step safety cap is explicitly a project evaluation cap: the original released MPC has `max_iter: null`.

DINO-WM keeps its own released Maze setup, including measured `(x,y,vx,vy)` proprioception. Its published 98% remains a published reference, with the input and evaluation-budget differences stated. A matched reproduction claim requires checking the released checkpoint under its full protocol; that reproduction has not passed here. The default training list therefore contains Fourier and LeWM, not an additional image-only DINO-WM run. An image-only DINO-WM ablation is optional and must be named separately; it is not a replacement for the published model. The source profile also records DINO-WM paper/code conflicts in optimizer and image size.

### Closed-loop arrival deadlines

Use `planner=sqp planner.starts=8 protocol=closed_loop protocol.replan_steps=5 protocol.feedback=true protocol.horizon_schedule=deadline` for the tested projected Gauss–Newton controller with per-observation TV-LQR. With `horizon_steps=25`, deadline mode plans 25,20,15,10,5 remaining primitive steps and starts a new segment if needed. This prevents replanning from repeatedly postponing the arrival time. It requires a replanning interval that is a multiple of the action block; feedback still runs every primitive step. Each environment maintains its own deadline, causal observation history and warm-start suffix. `horizon_schedule=fixed` remains the default for existing protocols. Report this schedule as part of the controller, and compare on the same saved tasks and success checks.


### Dynamics ablations and official PointMaze

Use `dynamics_type=linear|bilinear` with `model_family=fourier`; this retains the complete Fourier data/training/planning interface and changes only the primitive transition. Record the variant and predictor parameter count from `training_metadata.json`. Linear uses 37,248 dynamics parameters for a 192-dimensional latent and two controls; bilinear uses 110,976. These ablations intentionally have fewer parameters than Fourier, rather than adding unused or compensating networks. Each complete five-step forecast starts from an actual observation, with endpoints 5,10,15 for history three. TwoRoom retains its history-one paper exception. Shared initialization, split, seed, epoch budget and CEM tasks/settings should match the Fourier reference.

The official OGBench `pointmaze-large-navigate-v0` comparison uses the complete 1,000-episode visual training export and its separate 100-episode validation export. All models consume pixels only; Fourier/linear/bilinear receive commanded rates while LeWM/DINO-WM retain native actions. Use the validation artifact for the saved held-out evaluation task list and checkpoint-owned training action normalization. The project protocol uses 10 epochs for every model, training history three, evaluation history one, 50 saved tasks, offset25, budget50 and CEM300/30/30. LeWM and DINO-WM retain their source architecture and optimizer settings. This is a project benchmark, separate from DINO-WM's force U-maze and its reported result; the validation collection is not an additional independent test set.

The current Maze setups are intentionally separate:

| Dataset / model | Predictor inputs during training | Planning observation context | Training / status |
|---|---|---|---|
| Official OGBench / Fourier, linear, bilinear | One current image latent per forecast; commanded rate = native displacement ×0.2 /0.1 | One frame | 10 epochs, seeds42/43/44 |
| Official OGBench / LeWM | Up to three image latents at physical spacing5, with learned action embeddings | One primary; up to three in the paired context check | 10 epochs, seeds42/43/44 |
| Official OGBench / DINO-WM | Three image representations and action embeddings; frozen DINO backbone; no measured state | One primary; up to three in the paired context check | 10 epochs, seeds42/43/44 |
| Released force U-maze / Fourier, linear, bilinear | Concatenated three adjacent image latents, exact primitive history shift; native force commands; no action embedding | Up to three adjacent actual frames | 100 epochs, seeds42/43/44; queued behind allocated acceptance |
| Released force U-maze / LeWM | Original three-frame image transformer with learned action embeddings; native force commands | Up to three causal frames | 100 epochs, seeds42/43/44; queued behind allocated acceptance |
| Released force U-maze / original DINO-WM | Released image/action/proprioception encoders, including measured position and velocity | Released three-frame configuration | Published reference / released checkpoint; no new image-only training |

All newly trained models use 224px images. Fourier and LeWM use batch128, AdamW5e-5, weight decay1e-3 and MSE+0.09SIGReg. The trained OGBench DINO-WM baseline uses its own frozen-backbone loss, batch32, AdamW5e-4 and zero weight decay. OGBench evaluation uses held-out validation starts/goals with offset25 and budget50; force U-maze uses its separate 50 saved random-state tasks, seed99, CEM300/30/10 and a project cap250. Neither those tasks nor that cap establish a reproduction of the published DINO-WM98%. Initial missing history repeats the earliest available observation, without reading future frames. Current acceptance results establish pipeline correctness, not convergence or success of the new concatenated-state models.

The OGBench context check evaluates each completed LeWM/DINO-WM checkpoint twice: the original one-observed-frame evaluation and a separate three-frame variant. Dataset, exact50 tasks, reset/planner seeds, normalization, CEM300/30/30, horizon25, replanning25 and budget50 stay fixed. Only `history_len` and the output run name change. Context starts with one actual frame and grows causally; additional history affects later replans. Save the extra run under `_cem_history3`, with `context_comparison.json` reporting paired gains/losses, success rates and median planning time. The configured planner RNG seed is shared; sequential random draws can diverge after different episode lengths. Private campaign workers attach this evaluation to the existing allocation after the original evaluation, without retraining or another GPU job.
