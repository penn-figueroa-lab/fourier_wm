# Fourier World Model

Fourier latent dynamics and the supplied JAX SQP solver, integrated through
Stable World Model (SWM) interfaces. The imported `stable-worldmodel/`
tree is unchanged.

## Configuration

Run commands from the project root:

```text
configs/
├── train/training.yaml  # Fourier model; inherits SWM LeWM defaults and Cube data
└── plan/sqp.yaml        # our solver parameters
```

Cube task/data YAMLs are loaded directly from SWM; there are no local copies of
Cube/PushT task YAMLs or a `configs/data/` directory.
The configuration search path is relative: `file://./stable-worldmodel`.

## Install

Python >=3.10:

```bash
python -m pip install -e './stable-worldmodel[train,env,format]'
python -m pip install -e '.[planning,test]'
```

Planning uses JAX on CPU and Torch on CUDA. Set these before training/evaluation:

```bash
export JAX_PLATFORMS=cpu
export MUJOCO_GL=egl
export OMP_NUM_THREADS=4
```

## Cube expert data

SWM's original Cube training and planning configurations select
`ogbench/cube_single_expert.h5`. Install the original file at
`$STABLEWM_HOME/datasets/ogbench/cube_single_expert.h5`
(default cache: `~/.stable_worldmodel`).

Data source: [LeWM authors' original archive](https://huggingface.co/datasets/quentinll/lewm-cube/blob/02a19a67a0dc8c9d6215f89c19e0a597691e152a/cube_single_expert.tar.zst),
revision `02a19a67a0dc8c9d6215f89c19e0a597691e152a`.
Archive SHA256:
`3725d6a01abd492164441ef0a27e588f52b94a118fab56b96987b1a34a6c2600`.
The HDF5 contains 10,000 episodes and 2,010,000 observations, with 224x224 RGB
images and five primitive action dimensions. This is different from the prior
official play NPZ conversion. Matching the released data and public configuration
does not establish the exact data snapshot used for a published table.

## Train

```bash
.venv/bin/python -m fourier.training --config-name training \
  output_model_name=fourier_cube_expert
```

The shared configuration inherits SWM's LeWM optimizer, batch128, 100 epochs,
frameskip5, four sampled observations, ViT-tiny/patch14 trained from scratch,
SIGReg defaults, and seeded random 90/10 clip split. It replaces the entire model
mapping with FourierWorldModel/FourierDynamics. Primitive action width is inferred
from the dataset. Normalized raw actions pass through an Identity action encoder;
visual projector/prediction projection also remain Identity.

The supplied dynamics and loss computations are unchanged: encode four sampled
observations, apply five primitive transitions between adjacent observations,
and supervise the three endpoints. SIGReg and endpoint loss jointly update the
encoder and dynamics. Two loader workers and single-device training are retained.
Training keeps the final batch; unchanged validation code drops its final partial
batch. This is not an exact architectural/validation reproduction of LeWM.

For a small training/validation check, use a distinct output name:

```bash
.venv/bin/python -m fourier.training --config-name training \
  output_model_name=fourier_cube_expert_smoke \
  trainer.max_steps=200 trainer.limit_val_batches=32
```

Logs, resolved configuration and run metadata are stored under `artifacts/training/<run>/`.
All model checkpoints are stored under the Git-ignored `checkpoint/<run>/`:
`weights_epoch_N.pt`, `weights.pt` and `config.json` for inference;
`resume/last.ckpt` for full Lightning optimizer/scheduler/loop state.
The `checkpoint_dir: checkpoint` setting is relative to the project root.
Resume with the same run/configuration and `resume_from=<last.ckpt>`;
use a new output name and `resume_from=null` for a fresh run.

## Plan

Use SWM's original Cube task configuration and replace its solver group:

```bash
.venv/bin/python stable-worldmodel/scripts/plan/eval_wm.py \
  --config-dir configs --config-name cube \
  '~solver' '+plan@solver=sqp' \
  policy="${PWD}/checkpoint/fourier_cube_expert/weights.pt" \
  +plan_config.history_len=1 +plan_config.warm_start=true
```

Removing the upstream solver group prevents CEM fields from merging into SQP.
This uses the original Cube environment, expert dataset, seed42, 50 start/goal
pairs, goal offset25, 50-action budget, horizon5 blocks, and execution of all
five blocks (25 primitive actions) before replanning. SWM writes metrics and one
video per case alongside the selected checkpoint. Success uses SWM's 4cm cube
position criterion; initial states already within that threshold should be
reported separately.

For a two-case pipeline check, add `eval.num_eval=2` and use the smoke checkpoint.
This verifies execution, not learned performance.

The requested final comparison uses evaluation seeds 0, 1 and 2, with 50 cases
per seed (150 total), keeping the same trained checkpoint and planning protocol.
These are evaluation sampling seeds, not three independently trained models.
For metrics-only checks, call the same SWM interfaces with `World.evaluate(video=None)`;
no additional reports or video files are needed.

## Modules and validation

- `fourier.predictor`: supplied Fourier dynamics.
- `fourier.training`: supplied endpoint/SIGReg losses and existing SWM training glue.
- `fourier.world_model`: thin LeWM encode/rollout adapter; unpacks primitive controls.
- `fourier.sqp`: supplied projected damped Gauss-Newton solver and SWM adapter.

SQP retains the terminal mean-MSE objective, action slew penalty, box projection,
damping, five-point step search, and repeat-last warm start. Its adapter converts
between training and evaluation action normalization without changing physical
controls. A CEM comparison is a comparison of full planner configurations, since
the objectives and constraints also differ.

```bash
JAX_PLATFORMS=cpu .venv/bin/python -m pytest -q tests
```

Tests check endpoint gradients, the exact action Jacobian, inherited rollout,
JAX/Torch objective agreement, solver coordinates, configuration composition, and
the supplied source definitions. The hash fixture is versioned with the tests.

## Previous play-data run

The prior batch128 run completed 100 full epochs / 767,200 updates.
Its checkpoint, complete configuration, metrics, and 50-case videos remain in
their original run directories. The final seed42 SQP result was 24/50 (48%):
20/20 initially within 4cm and 4/30 initially outside. Those cases came from the
first ten play validation trajectories and are not comparable to expert-data
evaluation. Its resolved historical configuration remains with the old run
artifacts; new evaluations load SWM's original Cube configuration directly.

The two temporary configuration migration archives were deleted after the old
training and final evaluation were verified. No training logs, checkpoints,
dataset files, result files, or videos were deleted.

The supplied algorithm provenance is checked by `tests/original_hashes.json`.
Upstream attribution and license are retained in `stable-worldmodel/LICENSE`.
