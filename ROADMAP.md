# Roadmap — RealPDE Track 2 LTTTA

## Current State

- The repository layout, submission interface, and bounded controller (`skip_update` / `recalibrate` / `update_adapter`) are implemented and run end-to-end via `python3 scripts/local_eval.py` and package cleanly via `scripts/make_submission.py`.
- All three real-finetuned baseline checkpoints (CNO, Transolver, fp16 FNO) are downloaded and load correctly through `load_baseline.py`.
- **Phase 1 is done.** `data/train_real.tar.gz` is extracted and inspected (82 independent `{re}_{aoa}.h5` trajectories), a deterministic chronological/whole-trajectory splitter exists (`scripts/split_real_data.py`), and real frozen-baseline subscores for CNO/Transolver/fp16 FNO have been recorded on 11 genuinely held-out real trajectories. See [`docs/real_baseline_results.md`](docs/real_baseline_results.md) for full numbers and methodology. Headline: fp16 FNO wins on all five subscores and is ~7-8x faster on CPU than CNO/Transolver, contradicting the earlier "default to CNO" assumption.
- A first Phase 2 probe (untuned `policy.yaml` adapter, `mode: rule`, against fp16 FNO on the same held-out trajectories) scored *worse* than the frozen baseline on all five subscores — the controller's template thresholds are not a free win on real data and need real tuning before shipping. `data/train_sim.tar.gz` remains untouched (not needed yet, disk-constrained).
- No submission has ever been made to Codabench; no leaderboard score exists at any point in this project's history.

## Phase 1 — Done

Completed (full detail in [`docs/real_baseline_results.md`](docs/real_baseline_results.md)):

1. Inspected `train_real.tar.gz` (`tar -tzf` + one extracted sample) and confirmed the layout: 82 files, each an independent trajectory named `{re}_{aoa}.h5` with `u`/`v` fields at native `64x128`, ~868 frames (one short file at 282 frames).
2. Extracted the full archive under `data/train_real/` (6.9 GB; disk headroom stayed at 25-29 GB free throughout, well above the 5 GB stop threshold). `train_sim.tar.gz` was deliberately left untouched.
3. Wrote `scripts/split_real_data.py`: a deterministic, non-random, whole-trajectory split (sorted by `(re, aoa)`, held-out = the high-Re tail of that order), producing an 11-trajectory (433-step) held-out validation slice and a manifest (`outputs/real_data_split.json`).
4. Ran the unmodified CNO, Transolver, and fp16 FNO checkpoints (`mode: fixed` / `fixed_action: skip_update`) through the real streaming loop on those 11 held-out trajectories and recorded all five official subscores for each — the honest reference point that did not exist before.
5. Per-step timing was measured directly on real trajectory lengths (42 steps/trajectory typical, CPU-only on the dev machine, 0.5-4.2s/step depending on model); `time_budget_s: 60.0` in `policy.yaml` is comfortably above all three, so the per-trajectory adaptation budget is not the binding constraint (see the results doc's CPU-vs-platform-GPU timing caveat).

## Phase 2 — Medium-term (open)

1. ~~Train and validate the anchored residual adapter~~ — a first untuned run (`mode: rule` defaults) was tried against fp16 FNO on the Phase 1 held-out split and came out net-negative on all five subscores (see `docs/real_baseline_results.md`). **Still open:** an actual tuning pass — sweep `err_low`/`err_high`/`adapt_lr`/`adapt_steps`/`anchor_lambda`/`adapt_scope` against this same held-out split until `update_adapter` genuinely beats the frozen FNO baseline (or determine it structurally can't, and simplify the controller instead).
2. Tune `policy.yaml`'s remaining thresholds (`ema_beta`, `calib_momentum`, `sps_k` and its clamps) against the real validation split from Phase 1, replacing template defaults with values justified by data.
3. ~~Decide on a base model~~ — Phase 1's real comparison points clearly at fp16 FNO (wins every subscore, fastest per-step, fits the 256 MB cap via the existing fp16 packer). Re-confirm this holds after adapter tuning before finalizing the submission.
4. Package the tuned solution with `scripts/make_submission.py`, verify the extracted-size and re-import checks pass, and make an actual submission to Codabench; record the real leaderboard result (subscores and `final_score`) once it comes back.
5. The Phase 2 probe above is exactly the "adapter underperforms frozen baseline" signal item 1 anticipated — already acted on as the reason to tune before shipping, rather than assuming the adapter is net-positive by construction.

## Phase 3 — Stretch

1. Try `mode: llm` for real against the organizer gateway on a held-out real trajectory, and compare its action choices and resulting subscores against the `mode: rule` policy on the same data — not just confirm the fallback path works.
2. Compare final results against whatever the upstream [agentic_LTTTA](https://github.com/PgUpDn/agentic_LTTTA) repository itself reports, to understand how much of any gain comes from the controller idea versus this port's specific tuning.
3. Explore whether `adapt_scope: all` (full-parameter adaptation) ever beats the default `bn`-only (norm-layer) scope on real data, given enough anchor regularization to control drift.
4. Write a short, honest results write-up (what worked, what didn't, real subscores, leaderboard rank) once a submission has actually scored — useful both as a portfolio artifact and as a record for future track attempts.

## Success Metrics

- **Phase 1 done** ✅ when: the real archive is extracted and its layout documented, a chronological trajectory-level train/val split exists, and all five official subscores are recorded for all three frozen baselines on real held-out trajectories (numbers in a tracked file, not just terminal output). All satisfied — see `docs/real_baseline_results.md`.
- **Phase 2 done** when: the *tuned* adapter's five subscores on the same real held-out trajectories are recorded side-by-side with the Phase 1 frozen-baseline numbers and genuinely beat them, a submission has been made to Codabench, and an actual leaderboard `final_score` (and rank, if visible) has come back. (A first untuned adapter probe is recorded and was net-negative — tuning is still required to meet this bar.)
- **Phase 3 done** when: an `mode: llm` real-gateway run has been compared against `mode: rule` on the same real trajectories with recorded subscores, and a short results write-up exists summarizing what was tried and what the real (not synthetic) numbers showed.
