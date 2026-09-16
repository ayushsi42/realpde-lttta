# Roadmap — RealPDE Track 2 LTTTA

## Current State

- The repository layout, submission interface, and bounded controller (`skip_update` / `recalibrate` / `update_adapter`) are implemented and run end-to-end via `python3 scripts/local_eval.py` and package cleanly via `scripts/make_submission.py`.
- All three real-finetuned baseline checkpoints (CNO, Transolver, fp16 FNO) are downloaded and load correctly through `load_baseline.py`.
- `data/train_real.tar.gz` (~7.4 GB) and `data/train_sim.tar.gz` (~8.2 GB) are downloaded but **not yet extracted or inspected** — their internal file layout, trajectory count, and naming conventions are still unknown.
- **Zero evaluation has happened on real held-out PIV data.** Every number produced so far (Rel-L2/TKE/MVPE/Time/SPS via `scoring.py`) comes from the two tiny bundled synthetic trajectories in `data/example/`, which exist purely for shape/plumbing checks.
- `policy.yaml`'s thresholds are hand-set defaults from the upstream template, not tuned against any real validation signal.
- No submission has ever been made to Codabench; no leaderboard score exists at any point in this project's history.

## Phase 1 — Near-term (weeks)

1. **Inspect the real archive safely before extracting.** `tar -tzf data/train_real.tar.gz | sed -n '1,80p'` to see the file layout, naming, and per-trajectory structure without committing to ~7 GB of disk I/O blind. Do the same for `train_sim.tar.gz` if simulated pretraining data is needed for comparison.
2. **Extract under `data/train_real/`** (and `data/train_sim/` if needed), confirm disk space and extraction integrity, then remove the redundant `.tar.gz` only after `sha256sum`/spot-check confirms the extracted files are intact.
3. **Write a chronological (not random) train/val trajectory splitter.** Split by whole trajectory, in time order, so no window from a validation trajectory can leak into anything used to pick thresholds — random window-level splits would silently violate the causal contract this competition is testing.
4. **Establish frozen-baseline numbers first**, before touching the adapter further: run the unmodified CNO, Transolver, and fp16 FNO checkpoints (no adaptation, `mode: fixed` / `fixed_action: skip_update`) through the real streaming loop on the held-out real validation trajectories, and record all five official subscores (Rel-L2, TKE, MVPE, Time, SPS) for each. This is the honest reference point every later adaptation claim must beat.
5. Confirm the local evaluator's per-step timing and the 10-minute wall-clock budget are realistic on real trajectory lengths (not just the 4-step synthetic example) — real trajectories are almost certainly longer, and `time_budget_s` in `policy.yaml` (currently 60s per trajectory) needs sanity-checking against that.

## Phase 2 — Medium-term

1. Train and validate the anchored residual adapter properly: confirm `update_adapter`'s few-SGD-step / anchor-lambda design actually improves Rel-L2/TKE/MVPE over the frozen baseline on held-out real trajectories, not just that it runs without crashing.
2. Tune `policy.yaml`'s thresholds (`err_low`, `err_high`, `ema_beta`, `adapt_lr`, `adapt_steps`, `anchor_lambda`, `calib_momentum`, `sps_k` and its clamps) against the real validation split from Phase 1, replacing the current template defaults with values that are actually justified by data.
3. Decide on a base model (CNO vs. Transolver vs. fp16 FNO) for the shipped submission based on the real Phase 1 baseline comparison plus the 256 MB size cap, rather than defaulting to CNO because it's smallest.
4. Package the tuned solution with `scripts/make_submission.py`, verify the extracted-size and re-import checks pass, and make an actual submission to Codabench; record the real leaderboard result (subscores and `final_score`) once it comes back.
5. If the first submission underperforms the frozen-baseline numbers from Phase 1 on any subscore, treat that as a concrete signal to revisit the controller's thresholds or action space rather than assuming the adapter is net-positive by construction.

## Phase 3 — Stretch

1. Try `mode: llm` for real against the organizer gateway on a held-out real trajectory, and compare its action choices and resulting subscores against the `mode: rule` policy on the same data — not just confirm the fallback path works.
2. Compare final results against whatever the upstream [agentic_LTTTA](https://github.com/PgUpDn/agentic_LTTTA) repository itself reports, to understand how much of any gain comes from the controller idea versus this port's specific tuning.
3. Explore whether `adapt_scope: all` (full-parameter adaptation) ever beats the default `bn`-only (norm-layer) scope on real data, given enough anchor regularization to control drift.
4. Write a short, honest results write-up (what worked, what didn't, real subscores, leaderboard rank) once a submission has actually scored — useful both as a portfolio artifact and as a record for future track attempts.

## Success Metrics

- **Phase 1 done** when: the real archive is extracted and its layout documented, a chronological trajectory-level train/val split exists, and all five official subscores are recorded for all three frozen baselines on real held-out trajectories (numbers in a tracked file, not just terminal output).
- **Phase 2 done** when: the tuned adapter's five subscores on the same real held-out trajectories are recorded side-by-side with the Phase 1 frozen-baseline numbers, a submission has been made to Codabench, and an actual leaderboard `final_score` (and rank, if visible) has come back.
- **Phase 3 done** when: an `mode: llm` real-gateway run has been compared against `mode: rule` on the same real trajectories with recorded subscores, and a short results write-up exists summarizing what was tried and what the real (not synthetic) numbers showed.
