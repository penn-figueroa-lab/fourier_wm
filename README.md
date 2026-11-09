# Fourier World Model

Fourier latent dynamics and the supplied JAX SQP solver, integrated through
Stable World Model (SWM) interfaces. Local storage defaults disable persistent
evaluation results, videos and logs.

## Configuration

Run commands from the project root:

```text
configs/
├── model/fourier.yaml   # task-independent LeWM/Fourier architecture
├── train/training.yaml  # inherits SWM LeWM training defaults
└── plan/sqp.yaml        # legacy FourierWorldModel solver parameters
```

Cube/PushT task and data YAMLs are loaded from SWM; there are no local task
copies or configs/data directory. Relative search paths are file://./configs
and file://./stable-worldmodel. Select the task explicitly when training;
the upstream default task is PushT.

## Install

Python >=3.10:

```bash
python -m pip install -e './stable-worldmodel[train,env,format]'
python -m pip install -e '.[planning]'
```

Planning uses JAX on CPU and Torch on CUDA:

```bash
export JAX_PLATFORMS=cpu
export MUJOCO_GL=egl
export OMP_NUM_THREADS=4
```

## Cube expert data

SWM's Cube configurations select ogbench/cube_single_expert.h5. Install the
original file at $STABLEWM_HOME/datasets/ogbench/cube_single_expert.h5
(default cache: ~/.stable_worldmodel).

Data source: [LeWM authors' original archive](https://huggingface.co/datasets/quentinll/lewm-cube/blob/02a19a67a0dc8c9d6215f89c19e0a597691e152a/cube_single_expert.tar.zst),
revision 02a19a67a0dc8c9d6215f89c19e0a597691e152a.
Archive SHA256:
3725d6a01abd492164441ef0a27e588f52b94a118fab56b96987b1a34a6c2600.
The HDF5 contains 10,000 episodes and 2,010,000 observations, with 224x224 RGB
images and five primitive action dimensions. Matching the released data and
public configuration does not establish the exact snapshot used in a published table.

## Train

```bash
.venv/bin/python -m fourier.training --config-name training \
  scripts/train/config/data@data=ogb \
  output_model_name=fourier_cube_expert
```

The configuration inherits SWM's LeWM optimizer, batch128, 100 epochs,
frameskip5, four sampled observations, ViT-tiny/patch14 trained from scratch,
SIGReg defaults, and seeded random 90/10 clip split. The upstream LeWM wrapper
uses a FourierPredictor that unpacks each action block and calls the original
FourierDynamics. The visual projector remains SWM's 192 → 2048 → 192 MLP with
BatchNorm and GELU; action_encoder and pred_proj are Identity. Primitive action
width is inferred from the dataset.

Training calls SWM's lejepa_forward through LeWM.predict. Five primitive
transitions connect adjacent sampled observations, and the three endpoints
are supervised. Prediction loss plus 0.09 × SIGReg jointly update the encoder,
visual projector and Fourier dynamics. The supplied Fourier dynamics and SQP
definitions remain unchanged. Six loader workers with prefetch factor3 are
used. Training drops its final partial batch; validation includes its final
partial batch. The Cube expert split has 1,638,001 training clips
(12,796 updates per epoch at batch128) and all 181,999 validation clips
(1,422 batches). The Fourier predictor and identity prediction projection
differ from the upstream LeWM default architecture.

For a small training/validation check, use a distinct output name:

```bash
.venv/bin/python -m fourier.training --config-name training \
  scripts/train/config/data@data=ogb \
  output_model_name=fourier_cube_expert_smoke \
  +trainer.max_steps=200 +trainer.limit_val_batches=32
```

Training saves checkpoints under the Git-ignored checkpoint/<run>/ directory:
weights_epoch_N.pt, weights.pt and config.json for inference;
resume/last.ckpt for full Lightning model/optimizer/scheduler/loop state.
training_config.yaml and training_metadata.json retain data paths, normalization
and training settings; new resume checkpoints also embed this metadata.
CSV/JSONL logs, environment reports and Hydra output directories are disabled.
The checkpoint_dir setting is relative to the project root. Resume with the
same run/configuration and resume_from=<last.ckpt>; use a new output name and
resume_from=null for a fresh run.

## Plan

Use SWM's Cube task configuration with its default CEM solver for current
LeWM/Fourier checkpoints:

```bash
.venv/bin/python stable-worldmodel/scripts/plan/eval_wm.py \
  --config-name cube \
  policy="${PWD}/checkpoint/fourier_cube_expert/weights.pt" \
  +plan_config.history_len=1 +plan_config.warm_start=true
```

This uses the Cube environment, expert dataset, seed42, 50 start/goal pairs,
goal offset25, 50-action budget, horizon5 blocks, and execution of all five
blocks (25 primitive actions) before replanning. Results print to the terminal;
no evaluation result file, video or Hydra log is saved by default. Success uses
SWM's 4cm cube-position criterion; initially solved cases should be reported
separately. Enable persistence only when requested with output.save_results=true
and/or output.save_video=true.

The repository's SWMSQPSolver currently targets the legacy FourierWorldModel
interface. Current LeWM/Fourier checkpoints require an adapter that accesses
predictor.dynamics and reads action statistics from training_metadata.json.
The milestone SQP, multistart and TV-LQR experiments use in-memory adapters;
those adapters are not included in this CLI.

For a two-case pipeline check, add eval.num_eval=2 and use the smoke checkpoint.
This verifies execution, not learned performance. Evaluation seeds 0, 1 and 2
with 50 cases each are sampling seeds, not three independently trained models.
For metrics-only checks, use World.evaluate(video=None).

## Modules and validation

- fourier.predictor: supplied Fourier dynamics.
- fourier.training: endpoint/SIGReg losses and SWM training integration.
- fourier.world_model: FourierPredictor block adapter and legacy FourierWorldModel.
- fourier.sqp: supplied projected damped Gauss-Newton solver and legacy SWM adapter.

SQP retains terminal mean-MSE, action slew penalty, box projection, damping,
five-point step search, and repeat-last warm start. Its legacy adapter converts
between training and evaluation action normalization. CEM and SQP compare full
planner configurations with different objectives and constraints.

Checks live outside this repository at /home/zmhuang/wm_checks/fourier/ on the
lab machine. Do not add tests, numerical checks or generated reports here.
Run external numerical/interface checks without persistent cache/log output:

```bash
JAX_PLATFORMS=cpu .venv/bin/python -B -m pytest -q -p no:cacheprovider \
  /home/zmhuang/wm_checks/fourier/test_integration.py -k 'not config_targets'
```

The external config_targets check still describes the legacy architecture;
the command above runs the six numerical and interface checks. Validate the
current configuration with the explicit Cube override shown above.
Dynamics/loss/SQP definition hashes are kept with those external checks.
Checkpoints and their reload/resume configuration are retained. Evaluation
results, videos, reports and logs are generated only when explicitly requested.
artifacts/data contains evaluation inputs rather than evaluation results.

Upstream attribution and license are retained in stable-worldmodel/LICENSE.
