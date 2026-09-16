# RealPDE Track 2 — LTTTA
*A bounded, timed test-time adaptation controller for streaming real-world PIV airfoil-wake forecasting*

[![python](https://img.shields.io/badge/python-3.10%2B-blue)](#requirements)
[![competition](https://img.shields.io/badge/competition-NeurIPS%202026-blueviolet)](#overview)
[![status](https://img.shields.io/badge/status-in--development-yellow)](#current-status)

## Overview

This repository is an entry for **RealPDE Track 2: Long-Term Test-Time Adaptation (LTTTA)**, a NeurIPS 2026 competition track on adapting neural-operator PDE surrogates (CNO / FNO / Transolver) to streaming real-world particle image velocimetry (PIV) airfoil-wake flow data. The contract is deliberately hostile to naive fine-tuning: at each streaming step the model only ever sees the *previous* window's true target (delayed supervision, never the current one), every line of code inside `ttt_step` is wall-clock timed against a hard 10-minute per-trajectory cap, and the packaged submission archive must stay under 256 MB. The active solution is a bounded controller — adapted from the upstream [agentic_LTTTA](https://github.com/PgUpDn/agentic_LTTTA) baseline — that chooses, once per step, between skipping adaptation, recalibrating an output bias, or running a few anchored gradient steps, rather than blindly gradient-stepping on every noisy window.

## Key Features

- **Bounded action controller** (`skip_update` / `recalibrate` / `update_adapter`) driven by an EMA of the previous window's revealed relative-L2 error, so adaptation only fires when it's actually warranted.
- **Anchored adapter updates** — `update_adapter` runs a few SGD steps on the genuine previous `(input, target)` pair with an `anchor_lambda`-weighted L2 pull back toward the checkpoint's own weights, so a long run of updates can't drift arbitrarily far from the pretrained solution.
- **Offline-design / online-execution split** — all thresholds (`err_low`, `err_high`, `adapt_lr`, `anchor_lambda`, …) live in `src/solution/policy.yaml`, tuned offline and read deterministically at evaluation time under `mode: rule`.
- **Optional LLM-driven action selection** (`mode: llm`) through the organizer-provided gateway (auto-detected from injected `OPENAI_*` env vars), with a safe, deterministic rule-based fallback on any missing env, timeout, or malformed reply — the LLM can never crash the run.
- **Safe Prediction Score (SPS) bounds** returned every step, sized from a rolling residual EMA and converted to each channel's own normalized scale, instead of leaving the scorer to fall back on its default ±5% band.
- **Submission packaging pipeline** (`scripts/make_submission.py`) that stages the solution, injects a checkpoint, re-runs the local evaluator against the staged copy, checks the 256 MB extracted-size cap, and re-imports the zip exactly as the official evaluator does before calling it done.
- **fp16 checkpoint packer** (`scripts/pack_ckpt_fp16.py`, complex-tensor safe) so the 403 MB fp32 FNO checkpoint fits under the size cap.

## How It Works

Each streaming step follows the same causal loop: adapt on what was just revealed, then predict the current window without peeking at its answer.

```text
step t:
  1. reveal y_(t-1)          -- the ONLY new ground truth available this step
  2. score  ŷ_(t-1) vs y_(t-1)  -> rel-L2 error, EMA + slope
  3. controller decides one bounded action:
        err < err_low   -> skip_update      (do nothing)
        err > err_high  -> update_adapter   (few anchored SGD steps on
                                              cached x_(t-1), revealed y_(t-1))
        else            -> recalibrate      (cheap additive bias correction)
     [mode: llm asks the organizer gateway for the action instead, every
      llm_every-th step, with a rule-based fallback on any failure]
  4. predict ŷ_t = base(x_t) + calibration_bias   (no grad, no current target)
  5. emit SPS lower/upper bounds from a rolling residual estimate
  6. cache x_t for step t+1; everything above is wall-clock timed
```

`reset_ttt_state()` fires at every trajectory boundary: it restores the checkpoint's original weights and clears all controller/adapter state so each trajectory starts identically.

## Project Structure

```text
src/
  solution/                 Active submission (edit here)
    submission.py           Required entry point: get_ttt_model()
    policy.yaml             Controller thresholds / mode (rule | llm | fixed)
    README.md               How this maps onto the upstream agentic_LTTTA demo
  load_baseline.py          Loads CNO / FNO / Transolver checkpoints
  rpde_baselines/           Vendored organizer model implementations (do not edit)
  ttt_model.py              Optional convenience base class for the TTT interface

scripts/
  local_eval.py             Local streaming evaluator + real subscore printout
  make_submission.py        Stages, packages, and verifies outputs/submissions/*.zip
  scoring.py                Official subscore implementation (Rel-L2, TKE, MVPE, Time, SPS)
  pack_ckpt_fp16.py         fp32 -> fp16 checkpoint packer (complex-tensor safe)
  visualize_piv_window.py   Interactive PIV/window HTML explainer

data/
  example/                  Tiny tracked synthetic trajectories for shape/plumbing checks
  train_real.tar.gz         Full real training release (local, Git-ignored, not yet inspected)
  train_sim.tar.gz          Full simulated pretraining release (local, Git-ignored)

checkpoints/                Local, Git-ignored real-finetuned CNO/FNO/Transolver weights
docs/                       Interface contract, metrics, study guide, literature reviews, handoff notes
outputs/                    Locally generated ZIPs/visualizations (Git-ignored)
```

## Getting Started

### Requirements

- Python 3.10+ (developed against 3.12)
- `torch`, `numpy`, `h5py`, `pyyaml`
- `einops` only if you exercise the Transolver baseline directly (already present in the official evaluation image)
- No GPU required for the bundled smoke test; CUDA is used automatically if available

### Installation

```bash
git clone https://github.com/ayushsi42/realpde-lttta.git
cd realpde-lttta
pip install torch numpy h5py pyyaml
```

### Usage

```bash
# Run the active solution against the bundled tiny synthetic example data
python3 scripts/local_eval.py

# Run a specific submission directory / data directory / device explicitly
python3 scripts/local_eval.py --submission src/solution --data data/example --device cpu

# Regenerate the interactive PIV window explainer
python3 scripts/visualize_piv_window.py \
  --input data/example/test_real/5025_5.h5 \
  --output outputs/visualizations/piv_window_explorer.html

# Package a real checkpoint (set base_model: cno in policy.yaml first)
python3 scripts/make_submission.py --checkpoint sim_real_cno.pth
```

## Competition Contract

- **Delayed supervision**: `ttt_step(input_norm, prev_target_norm)` only ever receives the *previous* window's ground truth (`None` on a trajectory's first step) — the current target is never available.
- **Timing**: everything inside `ttt_step`, plus `reset_ttt_state()` (charged to the following step), is wall-clock timed and rolled into the `time_score` subscore; model construction and checkpoint loading are not timed. A separate, hard **10-minute** wall-clock limit covers the whole run per submission.
- **Size**: the extracted submission archive (checkpoint included) must stay under **256 MB** — only one checkpoint may ship; CNO (32 MB) and Transolver (48 MB) fit easily as fp32, FNO (403 MB fp32) needs the fp16 packer (~192 MB).
- **Tensor contract**: `(1, 20, 32, 64, 3)` normalized tensors, channels `[u, v, p]`; only `u, v` are scored on real data, `p` is zero-filled.
- **Isolation**: the submission runs in an isolated subprocess with no access to hidden ground truth on disk; only the values handed to `ttt_step` are usable.

## Current Status

Honest snapshot as of this writing:

- The repository is restructured into a standard `src/` / `scripts/` / `docs/` layout, and the active bounded controller (`src/solution/submission.py`) runs end-to-end.
- All three real-finetuned baseline checkpoints (CNO 31 MB, Transolver 48 MB, fp16 FNO 192 MB) have been downloaded and load correctly through `load_baseline.py`.
- The full real training archive (`data/train_real.tar.gz`, ~7.4 GB) and the simulated pretraining archive (`data/train_sim.tar.gz`, ~8.2 GB) have been downloaded but **have not yet been extracted or inspected**.
- **No evaluation against real held-out PIV trajectories has been run yet**, for either the frozen baselines or the adaptation controller. The only numbers produced so far come from `python3 scripts/local_eval.py` against the two tiny bundled synthetic trajectories in `data/example/` — that is a shape/plumbing smoke test, not a benchmark, and its subscores are explicitly not leaderboard-comparable (confirmed by re-running it: 4 steps over 2 trajectories, mean per-step time ~130 ms, and Rel-L2/TKE/MVPE/Time/SPS subscores computed by the real `scoring.py` on synthetic data only).
- No submission has been made to Codabench, and no leaderboard score exists.
- The adapter's anchor loss and SPS bound logic have been implemented and exercised on synthetic data, but `policy.yaml`'s thresholds (`err_low`, `err_high`, `adapt_lr`, `anchor_lambda`, …) are still defaults, not tuned against real validation trajectories.

See [ROADMAP.md](ROADMAP.md) for exactly what's next and why.

## Roadmap

The immediate priority is closing the gap between "runs on synthetic shape-check data" and "has a real, honest baseline-vs-adapted comparison on held-out real PIV trajectories," before any further controller tuning. See [ROADMAP.md](ROADMAP.md) for the full plan.

## Tech Stack

- **PyTorch** — model definitions, training loops, adaptation
- **CNO / FNO / Transolver** — vendored neural-operator baseline architectures (`src/rpde_baselines/`)
- **h5py / NumPy** — PIV trajectory I/O and array processing
- **PyYAML** — controller policy configuration
- **OpenAI-compatible client** — optional LLM action selection through the organizer gateway
- **Codabench** — competition submission platform (target)

## License

No `LICENSE` file is present in this repository. It began as an organizer-distributed starting kit for the RealPDE Track 2 competition (vendored baseline model code under `src/rpde_baselines/` is upstream competition code, not authored here), so a blanket license has not been added — check the competition's own terms before reusing or redistributing this code.

## Author

Ayush Singh — [GitHub](https://github.com/ayushsi42) · [LinkedIn](https://www.linkedin.com/in/ayush-singh-40539522b/) · ayushsingh73920@gmail.com
