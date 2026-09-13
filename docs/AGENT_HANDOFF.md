# Agent Handoff — RealPDE Track 2

## Project purpose

Build one Codabench submission for **RealPDE Track 2: Long-Term Test-Time Adaptation** on streaming real PIV airfoil-wake velocity fields. The model receives a 20-frame input window and predicts the next 20 frames. On the next call it receives the prior window's true target, which is legal delayed supervision for adaptation.

The active development principle is: **one evolving solution, with Git branches for tangential experiments**. Do not create a new permanent variant directory for each idea.

## Current repository layout

```text
src/
  solution/                 Active submission code — edit here
    submission.py           Required competition entry point
    policy.yaml             Controller/model configuration
  load_baseline.py          Loads CNO/FNO/Transolver checkpoints
  rpde_baselines/           Vendored organizer model implementations; do not edit
  ttt_model.py              Optional interface base class

scripts/
  local_eval.py             Local streaming evaluator
  make_submission.py        Creates outputs/submissions/submission.zip
  scoring.py                Official subscore implementation
  visualize_piv_window.py   Interactive PIV/window explainer generator
  pack_ckpt_fp16.py         fp32 → fp16 checkpoint packer

data/
  example/                  Small tracked smoke-test trajectories
  train_real.tar.gz         Full real training release; local and Git-ignored

checkpoints/                Local, Git-ignored checkpoints
docs/                       Interface, metrics, study guide, literature review
outputs/                    Local generated ZIPs/visualizations; Git-ignored
```

## Completed work

- Reorganized the supplied starting kit into the layout above.
- Moved active code to `src/solution/`; it is currently a bounded controller
  starting point derived from the organizer's agentic example.
- Added `docs/literature_review_test_time_adaptation.md`, rewritten as a
  beginner-friendly narrative with paper links.
- Added `scripts/visualize_piv_window.py`; it writes an interactive local HTML
  visualization under `outputs/visualizations/`.
- Downloaded and verified all three real-finetuned baseline checkpoints:

  ```text
  checkpoints/sim_real_cno.pth           31 MB,  7.96M parameters
  checkpoints/sim_real_transolver.pth    48 MB, 12.54M parameters
  checkpoints/sim_real_fno_fp16.pth     192 MB, 50.36M parameters
  ```

- Downloaded the full real training archive: `data/train_real.tar.gz`
  (about 7.39 GB decimal / 6.9 GiB). It has completed downloading but has not
  yet been inspected or extracted.
- Local smoke test and submission packaging worked after the restructure:

  ```bash
  python3 scripts/local_eval.py
  python3 scripts/make_submission.py
  ```

## Important competition facts

- Tensor shape inside `ttt_step`: `(1, 20, 32, 64, 3)`, channels `[u, v, p]`.
  Only `u, v` are scored; real-data `p` is zero-filled.
- Raw PIV is `64 × 128`, downsampled by taking every second row/column to
  `32 × 64`, then normalized before the submitted model sees it.
- At step `t`, only the previous target `y_(t-1)` is available. Never use the
  current hidden target `y_t` for the current prediction.
- Call `reset_ttt_state()` at each trajectory boundary; it must restore the
  checkpoint and clear all adaptation state.
- The uploaded extracted archive must be under 256 MB. Only one checkpoint is
  included. CNO and Transolver fit easily; only the provided fp16 FNO fits.
- Runtime in `ttt_step`, adaptation, controller logic, and reset all count.

## Current code caveat before packaging a real checkpoint

`scripts/make_submission.py --checkpoint ...` copies the selected file into
the ZIP as `model.pth`. Since that generic name no longer tells the loader
whether it is CNO, FNO, or Transolver, set `base_model` explicitly in
`src/solution/policy.yaml` before packaging:

```yaml
base_model: cno          # or transolver / fno
```

Start with CNO. Do not package all three checkpoints: together they exceed the
256 MB cap.

## Recommended next steps

1. Inspect the real-data archive without extracting blindly:

   ```bash
   tar -tzf data/train_real.tar.gz | sed -n '1,80p'
   ```

2. Extract it under `data/train_real/`, then remove the duplicate `.tar.gz`
   only after confirming extraction succeeds and disk space is healthy.
3. Read the resulting file layout and write a small chronological validation
   splitter/evaluator. Split by trajectory, not random windows, to avoid future
   leakage.
4. Establish frozen real-finetuned CNO, Transolver, and fp16 FNO baselines on
   the same held-out real trajectories. Compare Rel-L2, TKE, MVPE, SPS, and
   runtime; do not select from the tiny `data/example` set.
5. Implement the user's chosen method inside `src/solution/`. A strong first
   research baseline is a small, anchored residual adapter trained only on the
   previous revealed `(input, target)` pair, with a conservative update gate.

## Useful commands

```bash
# Current toy/smoke test
python3 scripts/local_eval.py

# Generate the visual explanation
python3 scripts/visualize_piv_window.py \
  --input data/example/test_real/5025_5.h5 \
  --output outputs/visualizations/piv_window_explorer.html

# Package CNO after setting `base_model: cno` in policy.yaml
python3 scripts/make_submission.py --checkpoint sim_real_cno.pth

# Confirm current repository state
git status
```

## Recent Git history

```text
999a290 Rewrite TTA literature review as a guided narrative
6a915e3 Fix literature review math rendering
b4a7a2f Reorganize project into standard ML layout
```
