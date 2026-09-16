# Real Frozen-Baseline Results (Phase 1)

Honest reference numbers for CNO, Transolver, and fp16 FNO on genuinely
held-out real PIV trajectories, obtained by actually running each frozen
checkpoint through the same streaming protocol `scripts/local_eval.py` uses
(prev-target revealed, current target hidden), scored with the real
`scripts/scoring.py`. Nothing here is estimated or fabricated.

## Data and split

- Source: `data/train_real.tar.gz`, extracted to `data/train_real/train_real/`
  (82 files, one PIV trajectory per file, named `{re}_{aoa}.h5`: Reynolds
  number and angle of attack in degrees, confirmed from the `re`/`aoa` scalar
  datasets inside each file). Native resolution `64 x 128`, `u, v` channels
  (float64 on disk), 868 native frames per full trajectory (one short file,
  `24150_20.h5`, has 282 frames).
- Split: `scripts/split_real_data.py`, a deterministic whole-trajectory split
  (never random, never window-level). Trajectories are sorted ascending by
  `(re, aoa)` -- the factorial-sweep design axis, since each file is an
  independent run with its own restarting time axis and there is no shared
  wall-clock timestamp across files to sort by. The held-out validation set is
  the contiguous high-Re tail of that order: **11 trajectories** with
  `re in {24150, 25425, 26700}`, listed below. The remaining 71 trajectories
  are the "train" side of the manifest (not used for anything in Phase 1,
  since no adapter training happened yet -- see Phase 2 stretch below).

  Held-out (11): `24150_0, 24150_10, 24150_15, 24150_20, 25425_0, 25425_10,
  25425_15, 25425_20, 26700_0, 26700_10, 26700_15`

  Manifest: `outputs/real_data_split.json`.

- Streaming protocol (fixed by the competition contract, mirrored from
  `scripts/local_eval.py`): `T_in = T_out = 20`, stride 20, spatial
  subsample `2x` (`64x128 -> 32x64`), channel 3 (`p`) zero-filled. This
  produced **433 streaming steps** across the 11 held-out trajectories
  (42 steps for each 868-frame trajectory, 13 for the one 282-frame
  trajectory).
- Normalization: the official `data/example/mean_std_real.pt` stats (the same
  ones the checkpoints were fine-tuned under), copied alongside the held-out
  symlink directory by `split_real_data.py`.

## Method

For each checkpoint, the *unmodified* active submission controller
(`src/solution/submission.py`) was run with `policy.yaml` set to
`mode: fixed`, `fixed_action: skip_update`, and `base_model: <cno|transolver|
fno>` pointing `model.pth` at the corresponding checkpoint under
`checkpoints/`. `skip_update` means no adaptation, no calibration bias, no
gradient steps ever run -- this is the genuine frozen checkpoint's forward
pass under the real streaming loop, exactly the "no touching the adapter"
reference point Phase 1 calls for. Evaluation used
`scripts/local_eval.py --data data/train_real_holdout --device cpu` (no GPU
available on this machine).

## Results

All five official subscores from `scripts/scoring.py`, computed over all 433
steps pooled across the 11 held-out trajectories (CPU wall-clock timing; see
caveat below):

| Model | rel_l2_score | tke_score | mvpe_score | time_score | sps_score | mean step time (ms) |
|---|---|---|---|---|---|---|
| CNO (fp32, 31 MB) | 95.245 | 73.189 | 96.009 | 29.403 | 27.604 | 4202.19 |
| Transolver (fp32, 48 MB) | 93.495 | 70.560 | 94.633 | 30.531 | 18.735 | 3774.01 |
| FNO (fp16, 192 MB) | 96.638 | 77.308 | 97.587 | 54.006 | 36.364 | 528.71 |

**Headline finding:** on this held-out real slice, fp16 FNO wins outright on
*every one* of the five subscores, and is roughly 7-8x faster per step than
CNO/Transolver on CPU. The project's earlier assumption ("start with CNO
because it's smallest") is not supported once real data is actually run
through the models; FNO looks like the stronger base-model choice for a
submission, pending further validation.

**Timing caveat:** these runs are CPU-only wall-clock (this machine has no
CUDA device); the official evaluation platform may use a GPU, so
`time_score` here is a lower bound on what the platform would report, not
necessarily what a real submission would score. The other four subscores
(rel_l2, tke, mvpe, sps) are hardware-independent (they depend only on the
model's numerical output), so those are directly meaningful.

## Phase 2 stretch attempt

Attempted (a): ran the *unmodified* `policy.yaml` default (`mode: rule`,
untuned `err_low: 0.035` / `err_high: 0.11` / `adapt_steps: 5` / `adapt_lr:
0.001` / `anchor_lambda: 0.01`, `adapt_scope: bn`) against the fp16 FNO
checkpoint -- the best frozen baseline above -- over the identical 11
held-out real trajectories (433 steps), so the bounded controller's
`skip_update` / `recalibrate` / `update_adapter` actions actually fire on the
genuine revealed-previous-target stream, exactly as `AgenticTTTModel` runs in
the real submission. Same command as Phase 1, just pointing at a submission
dir with default `mode: rule` from `src/solution/policy.yaml`, `base_model:
fno`, checkpoint `sim_real_fno_fp16.pth`.

| | rel_l2_score | tke_score | mvpe_score | time_score | sps_score | mean step time (ms) |
|---|---|---|---|---|---|---|
| FNO frozen (`skip_update`) | 96.638 | 77.308 | 97.587 | 54.006 | 36.364 | 528.71 |
| FNO + adapter (`mode: rule`, untuned) | 94.811 | 74.661 | 95.398 | 37.880 | 21.000 | 1960.34 |

**Result: the untuned adapter is net-negative on real data, on every single
subscore.** It is worse on accuracy (rel_l2/tke/mvpe all drop), much slower
(mean step time nearly 4x higher, since `update_adapter`'s few SGD steps plus
backward pass fire whenever the rolling error EMA crosses `err_high`), and
worse on SPS. This is exactly the concrete signal ROADMAP Phase 2 item 5
called for: the controller's default template thresholds are not a free win
on real data and should not be shipped as-is. The honest conclusion is that
`policy.yaml`'s `err_low`/`err_high`/`adapt_lr`/`anchor_lambda` need to be
tuned against this same held-out split (or the adapter path rethought)
*before* an adapted model can be justified over the plain frozen fp16 FNO
checkpoint for a real submission. That tuning sweep was not attempted here
and remains open in `ROADMAP.md`.

Not attempted: the `mode: llm` organizer-gateway smoke test -- skipped in
favor of finishing the adapter comparison above within the time available for
this pass; it remains open as a Phase 3 item.

## Reproduce

```bash
python3 scripts/split_real_data.py   # writes data/train_real_holdout/ + outputs/real_data_split.json
# then for each model, point base_model at cno / transolver / fno in a copy of
# policy.yaml with mode: fixed, fixed_action: skip_update, and model.pth
# symlinked to the corresponding checkpoints/sim_real_*.pth, then:
python3 scripts/local_eval.py --submission <that dir> --data data/train_real_holdout --device cpu

# Phase 2 adapter comparison: same setup but leave policy.yaml's default
# mode: rule (don't set mode: fixed), base_model: fno, checkpoint
# sim_real_fno_fp16.pth, same --data data/train_real_holdout.
```
