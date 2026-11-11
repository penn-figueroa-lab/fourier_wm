# Testing the Two-Room checkpoints

Use this checkout for current evaluation. The shared bundle contains `checkpoints/tworoom_fourier_s42`, `s43` and `s44`. Keep each directory together: `weights.pt`, `config.json` and `training_metadata.json` are required. The metadata contains that model's training normalization. Saved tasks and original results are in each directory's `evaluation/` folder.

The bundle also includes the frozen PARCC training code and evaluation overlay, preserving the historical experiment. This checkout has newer integration changes. Strict loading and deterministic encoder/predictor outputs have been checked against the original Two-Room code for all three seeds. Exact training resume requires the original frozen source and full resume checkpoint, rather than inference weights.

## 1. Check the weights without a dataset

From the repository root:

```bash
bash scripts/install.sh
source .venv/bin/activate
export CHECKPOINTS=/absolute/path/to/shared/checkpoints
export FOURIER_RUNS=/absolute/path/to/tworoom-evaluations

fourier-wm doctor --environment tworoom \
  checkpoint="$CHECKPOINTS/tworoom_fourier_s42/weights.pt"
```

This checks the upstream tree, strict loading, finite model tensors, command semantics and saved normalization. It does not train anything or measure planning success.

To watch the model control a native seeded task **without downloading data**, run:

```bash
python scripts/demo_tworoom.py \
  --checkpoint "$CHECKPOINTS/tworoom_fourier_s42/weights.pt" \
  --output "$FOURIER_RUNS/demo_s42"
```

Open `demo_s42/rollout.gif`: the current observation is on the left and the goal image on the right. This runs CEM with five-step replanning. For eight-start Gauss–Newton with TV-LQR, add `--planner sqp --feedback` and choose a new output directory. The demo uses a native reset task and records its settings; its outcome is not the saved 50-task paper score.

## 2. Prepare the original data

Use the [LeWM Two-Room release](https://huggingface.co/datasets/quentinll/lewm-tworooms/tree/6903a2de048b13819d812da0b4dd661290bc01e4), revision `6903a2de048b13819d812da0b4dd661290bc01e4`: `tworoom.tar.zst`, containing `tworoom.h5`. It has 10,000 heuristic trajectories and 920,809 stored observations. Keep the archive for provenance verification. A random collection is a different benchmark.

```bash
export DATA=/absolute/path/to/tworoom-data
fourier-wm prepare-data --environment tworoom \
  --source "$DATA/tworoom.h5" --archive "$DATA/tworoom.tar.zst" \
  --output "$DATA/train.json" --split train
```

For **exact saved-task replay**, copy `dataset/train.json` from the handoff to `$DATA/train.json` instead of creating a new manifest. Change only its `path` field to the absolute local location of `tworoom.h5`. Schema-v2 identities exclude that locator; content verification remains active. A newly prepared manifest may have different provenance fields and therefore a different identity. Do not edit an identity to force a match. The dataset bytes must match SHA-256 `129a36aa93ea0de488d2bcc876e396de9e3907bf66c6aae6394e542ef6a6d623`.

## 3. Replay the saved CEM comparison

```bash
fourier-wm eval --environment tworoom protocol=lewm_paper planner=cem \
  seed=42 run_name=tworoom_s42_paper_cem \
  checkpoint="$CHECKPOINTS/tworoom_fourier_s42/weights.pt" \
  dataset.manifest="$DATA/train.json" \
  tasks="$CHECKPOINTS/tworoom_fourier_s42/evaluation/tasks.json"
```

The preset fixes 50 tasks, goal offset 25, budget 50 primitive steps, horizon 25, replanning every 25, and CEM with 300 candidates, 30 elites and **10 iterations**. Evaluation context is one image. Recorded seed-42/43/44 results are 96%/94%/96%; these are historical results, not scores from this checkout's artifact check.

Repeat with another checkpoint and a new run name, keeping **evaluation seed 42 and the same task file**. Training seed belongs to the checkpoint; changing the evaluation seed changes planner randomness and fails the saved-task check. Reports go to `$FOURIER_RUNS/<run_name>/`: results, episodes, tasks, configuration and provenance. Existing directories are protected from overwrite. Add `output.save_video=true` for videos.

## 4. Test frequent replanning and feedback

```bash
fourier-wm eval --environment tworoom protocol=closed_loop planner=sqp \
  seed=42 run_name=tworoom_s42_gn_tvlqr \
  checkpoint="$CHECKPOINTS/tworoom_fourier_s42/weights.pt" \
  dataset.manifest="$DATA/train.json" \
  tasks="$CHECKPOINTS/tworoom_fourier_s42/evaluation/tasks.json" \
  protocol.num_eval=50 planner.starts=8 protocol.replan_steps=5 protocol.feedback=true \
  protocol.horizon_schedule=deadline
```

This uses eight projected Gauss–Newton starts, replans every five primitive steps and applies TV-LQR at every observation. `deadline` uses horizons 25, 20, 15, 10 and 5 within each 25-step segment. For CEM with the same project execution convention, use `planner=cem`, omit `planner.starts`, and set `protocol.feedback=false`; keep the tasks, replanning interval and horizon schedule identical.

Paper CEM follows upstream execution without wrapper clipping. Project closed-loop evaluation clips executed actions to native bounds; Gauss–Newton also constrains its search. Report these protocols separately. A short artifact or simulator check cannot establish that frequent replanning improves success.

## What these weights learned

One image; ViT-tiny encoder; LeWM projector; 192-dimensional latent; 768 Fourier frequencies. No action embedding or nonlinear prediction head. For native relative command `a`, Fourier receives `50 * a` in pixel/s, followed by the saved mean/scale normalization. Execution reverses this conversion.

Each training clip contains `o_t`, `o_(t+5)` and five intervening primitive actions:

```text
z = E(o_t)
for j = 0, ..., 4:
    z = F(z, u_(t+j))
loss = MSE(z, E(o_(t+5))) + 0.09 * SIGReg(encoded observations)
```

Gradients flow through all five steps and the encoder. Saved settings are `history_size=1`, `num_preds=1`, `num_steps=2`, `frameskip=5`, for 10 epochs. This is one endpoint loss per clip. The requested three-pair setup would also include `o_(t+10)` and `o_(t+15)`, supervising three teacher-forced five-step forecasts. These weights must not be labeled as that experiment. No objective or checkpoint changes silently during the handoff.
