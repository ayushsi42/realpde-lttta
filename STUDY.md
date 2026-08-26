# RealPDE Track 2 (LTTTA) — Study Guide

Personal study doc. Goal: understand the contest deeply enough to make real
design decisions, not just get a submission to run.

---

## 1. What the contest actually wants from you

**Track 2 = Long-Term Test-Time Adaptation (LTTTA)** on real airfoil-wake PIV
(Particle Image Velocimetry) flow data.

You are **not** training a model from scratch. Organizers give you pretrained
neural PDE surrogates (CNO / FNO / Transolver, optionally sim-pretrained or
sim+real-finetuned). Your job is to write an **online adaptation wrapper**
around a base model that is fed a live stream of windows and must keep
predicting well — and getting *better*, not worse — as the stream goes on,
using only information revealed so far.

### The streaming protocol, precisely

- Data arrives as **trajectories** (separate flow sequences). Each trajectory
  is a sequence of fixed-size windows: 20 input frames → 20 target frames,
  stride 20 (non-overlapping).
- At each step `k` you get `input_norm` (current 20 frames) and
  `prev_target_norm` (the **true** target of step `k-1`, i.e. genuinely
  revealed ground truth — but never the current step's target).
- You must return a prediction for the *current* window before you're allowed
  to see its answer. You only find out if you were right one step later, when
  it becomes `prev_target_norm`.
- At a trajectory boundary, `reset_ttt_state()` is called — you're expected
  to restore original checkpoint weights and wipe any adaptation state, so
  each trajectory starts from the same base model.

This is the key structural fact: **you're always one step behind on
feedback**, and adaptation has to be causal (only use `prev_input` +
`prev_target`, never peek at the current target). `submission_template.py`'s
`ReferenceTTTModel` shows the minimal legal loop:
1. If `prev_target_norm` given: one SGD step on `(cached prev_input, prev_target)`.
2. Predict current input, no grad, no current target.
3. Cache current input for next time.

### Why "long-term" matters — the real intellectual problem

A single gradient step per window sounds harmless. Repeated over a long
trajectory (or worse, across many trajectories if state isn't reset
correctly), naive per-step SGD on noisy single-window supervision can:
- **overfit to recent noise** in the PIV measurement (real data, not clean
  simulation — noisy by construction),
- **drift away from the pretrained solution** (catastrophic forgetting) since
  nothing anchors it back to the checkpoint except the trajectory reset,
- do this compounding over dozens/hundreds of steps in ways that don't show
  up if you only test on one or two windows.

This is *why* the competition ships `agentic_demo/`: a controller that
**chooses, per step, whether to adapt at all** (skip / recalibrate / full
update), rather than blindly gradient-stepping every single time. That's one
concrete philosophy for taming the long-term drift problem — not the only
one, but a strong hint about what the organizers think matters.

### The five things you're scored on

Every subscore is 0–100, higher is better, and they trade off against each
other:

| Subscore | What it measures | Where the tension is |
|---|---|---|
| `rel_l2_score` | Raw prediction accuracy (relative L2 on u, v) | baseline accuracy |
| `tke_score` | Turbulent kinetic energy consistency (variance over time of u, v) | matching *statistics*, not just pointwise values |
| `mvpe_score` | Mean velocity profile error at fixed wake-probe grid points | getting the *physically meaningful* locations right, not just global average |
| `time_score` | Mean per-step wall time vs. a numerical-solver reference (`t_numerical = 0.72896s`) | more adaptation = slower = lower score |
| `sps_score` | Safe Prediction Score: accuracy + *calibrated, tight* uncertainty intervals | wide intervals score safe but weak; narrow wrong intervals score zero |

`final_score` combines all five; the combination formula is **not
published**. So you can't just max one subscore — you have to reason about
the whole vector. This is itself worth sitting with: without knowing the
weights, robustness across all five is probably safer than maximizing one.

### The engineering contract (get this exactly right, it's unforgiving)

- `get_ttt_model(submission_dir, device)` — called once, NOT timed. Load
  checkpoint here.
- `reset_ttt_state()` — called at every trajectory start, IS timed (charged
  to the following step).
- `ttt_step(input_norm, prev_target_norm)` — called every step, FULLY timed
  (adaptation + controller logic + optional LLM calls + forward pass, all of
  it). Must return `(pred_norm, info)` with `info["adapt_loss"]` (float or
  `None`).
- Optional `info["lower"]` / `info["upper"]` for SPS: same shape as
  `pred_norm`, same normalized space, **both or neither**, and once you start
  returning them you must return them on *every* remaining step of the run —
  no lone key, no switching mid-stream (`local_eval.py` enforces this
  locally exactly like the real evaluator does).
- 256 MB extracted archive size cap (checkpoint included) — `FNO` at fp32 is
  403 MB and needs `pack_ckpt_fp16.py` to fit; `CNO` (32MB) and `Transolver`
  (50MB) don't.
- 10-minute wall-clock cap for the *entire* run (load + full stream).
- Sandbox: your code runs in an isolated subprocess; ground truth files are
  literally unreadable from disk (kernel-enforced) — you're never meant to
  need direct data access, only what's passed into `ttt_step`.

---

## 2. What I ran locally (already done, for reference)

Environment check: Python 3.12.2, torch 2.3.1, h5py 3.10.0, einops 0.8.2 — all
present, no install needed.

```bash
cp submission_template.py submission.py
python3 local_eval.py --submission .
```

Result (tiny synthetic `example_data`, NOT leaderboard-comparable, illustrative only):

```
4 steps over 2 trajectories, batch size 1
prediction stack shape (4, 20, 32, 64, 3)
mean per-step time: 83.73 ms
SPS bounds: none -> scorer default +/-5% band
rel_l2_score  76.500
tke_score     73.252
mvpe_score    86.472
time_score    74.687
sps_score      1.673
```

And the bundled agentic baseline:

```bash
python3 local_eval.py --submission agentic_demo
```

```
rel_l2_score  77.849
tke_score     79.279
mvpe_score    87.462
time_score    74.313
sps_score      1.918
```

**Notable observation to think about**: `sps_score` is near-zero in both
runs, using the scorer's *default* `±5%` interval band (neither submission
sets `info["lower"]/info["upper"]`). Look at the SPS formula in
`docs/metrics.md` / `scoring.py::aggregate_sps` — interval width is
normalized by a **frozen constant** `sigma_global = 0.0563870` (mean std of
u,v on the real training split). A `±5% of |prediction|` band is a
prediction-magnitude-relative width, completely decoupled from
`sigma_global`'s fixed physical scale — worth working out by hand *why* that
mismatch tanks the score, rather than me telling you. That's a genuine
design decision the template leaves on the table (see the commented-out
`lower`/`upper` block in `submission_template.py`).

**No baseline checkpoint was loaded in this run** — `TinyForecaster` is an
untrained residual conv net (near-identity/persistence prior). The real
baseline checkpoints (CNO/FNO/Transolver, sim-pretrained or sim+real
fine-tuned) live on the competition Google Drive, not shipped in this kit.
Getting one of those loaded via `load_baseline.py` and wired into
`ReferenceTTTModel` is the natural next hands-on step, once the study below
is further along — deliberately not doing that yet per your ask.

---

## 3. Study sources, mapped to what you need to understand

### A. Domain physics — PIV & wake turbulence
Goal: know what you're looking at when you see `u, v` fields and understand
why TKE and wake probes are meaningful things to score.
- **Particle Image Velocimetry (PIV)** — Wikipedia entry + any intro fluid
  mechanics course page is enough; you need "how is velocity measured
  optically" at a conceptual level, not the optics.
- **Kármán vortex street** (the classic airfoil/cylinder wake pattern) —
  Wikipedia has good visuals; this is almost certainly the flow regime the
  dataset captures.
- **Turbulent kinetic energy (TKE)** — any intro turbulence notes (search
  "turbulent kinetic energy definition variance mean flow"). The scorer's
  definition (`0.5 * (var_t(u) + var_t(v))`, variance over the 20 frames) is
  a simplified single-point TKE proxy — know the general concept, then read
  `scoring.py::kinetic_energy` yourself to see the simplification.

**Visual resources:**
- [The Beauty of Vortex Streets](https://www.comsol.com/blogs/the-beauty-of-vortex-streets) (COMSOL Blog) — real satellite photos + simulation animations of vortex streets forming behind islands. Best "oh, that's what this looks like" starting point.
- [A Stroll down Karman Street](https://www.ias.ac.in/article/fulltext/reso/010/08/0025-0037) (Resonance/IAS) — short, diagram-rich intro article.
- [Von Kármán Street Tutorial](https://basicairdata.eu/knowledge-center/computational-fluid-dynamics/von-karman-street-tutorial/) (BasicAirData) — CFD walkthrough with vortex-shedding animations, maps directly onto the wake-probe geometry used in `mvpe_rel_l2_per_sample`.
- [Understanding Turbulent Kinetic Energy](https://resources.system-analysis.cadence.com/blog/msa2021-understanding-turbulent-kinetic-energy-and-randomness) (Cadence) — plain-language, diagram-supported TKE intuition.

### B. Neural PDE surrogates — the three baseline architectures
Goal: know *what problem* each solves and *why* you'd choose one as your TTT
base, not derive the math.
- **Fourier Neural Operator (FNO)** — Li et al., "Fourier Neural Operator for
  Parametric Partial Differential Equations" (arXiv 2010.08895). Read
  abstract + Figure 1-2. Concept: learns a kernel in Fourier space, resolution
  -independent.
- **Convolutional Neural Operator (CNO)** — Raonić et al., "Convolutional
  Neural Operators for robust and accurate learning of PDEs" (arXiv
  2302.01178). Concept: bandlimiting + conv layers so it behaves like a
  proper operator across resolutions.
- **Transolver** — Wu et al., "Transolver: A Fast Transformer Solver for PDEs
  on General Geometries" (arXiv 2402.02366). Concept: attention over learned
  "slices" (physics-aware tokens) instead of raw grid points.
- After reading abstracts, look at the size/speed tradeoff you already have
  data for: CNO 32MB, Transolver 50MB, FNO 403MB(fp32)/201MB(fp16). Bigger
  isn't automatically better under a 10-minute wall-clock budget — connect
  this back to `time_score`.

**Visual resources:**
- [Zongyi Li's FNO blog](https://zongyi-li.github.io/blog/2020/fourier-pde/) — written by the original FNO author, ~10 min, clean architecture diagrams. Best single source for FNO intuition.
- [Neural Operators: FNO, AFNO, CNO, UNO](https://www.emergentmind.com/topics/neural-operators-fno-afno-cno-uno) (Emergent Mind) — side-by-side visual comparison of the operator family, useful for seeing where CNO sits relative to FNO.
- [Operator Learning: CNO explained](https://medium.com/@bogdan.raonke/operator-learning-convolutional-neural-operators-for-robust-and-accurate-learning-of-pdes-ebbc43b57434) (Medium, by a CNO paper author) — includes the data-efficiency comparison charts vs. FNO.
- [neuraloperator docs — FNO theory guide](https://neuraloperator.github.io/dev/theory_guide/fno.html) — clean official diagrams, good cross-check. No polished visual write-up exists yet for Transolver specifically — the arXiv paper's own figures are the best visual source there.

### C. Test-Time Adaptation (TTA) — the actual subject of the contest
Goal: build a mental model of *why* naive per-step gradient updates on a
streaming target are risky, and what the standard mitigations are, so you can
evaluate (not necessarily copy) the `agentic_demo` controller's approach.
- **TENT** — Wang et al., "Tent: Fully Test-Time Adaptation by Entropy
  Minimization" (arXiv 2006.10726). Foundational TTA paper; read for the
  general idea of adapting only specific parameters (e.g. batch-norm) rather
  than the whole network.
- **Continual / Online Test-Time Adaptation** — search "continual test-time
  adaptation catastrophic forgetting" (e.g. Wang et al. CoTTA, arXiv
  2203.13591). This is the closest match to "long-term": read for how they
  handle *drift over a long stream* specifically — stochastic weight
  restoration, teacher-student anchoring, etc. You don't need to implement
  their exact method; you need the vocabulary and the failure modes they
  describe.
- **Catastrophic forgetting** (general continual learning concept) — any
  intro continual-learning survey abstract. Understand it as a concept
  independent of TTA before mapping it onto this stream setting.
- Then re-read `agentic_demo/README.md` and skim `agentic_demo/submission.py`
  / `agentic_demo/policy.yaml` with this vocabulary in hand: identify which
  known TTA failure mode each of its three actions (`skip_update` /
  `recalibrate` / `update_adapter`) is defending against.

**Visual resources:** weakest area for visuals — nothing Distill-caliber
exists specifically for TTA yet, so lean on these plus the papers' own figures.
- [Catastrophic Forgetting — Thomas Zilliox](https://medium.com/@thomas.zilliox/catastrophic-forgetting-or-the-challenge-of-continuous-learning-1278a1179811) (Medium) — accessible, diagram-supported intro to the core failure mode; good on-ramp before TENT itself.
- [TENT official repo](https://github.com/DequanWang/tent) + the [ICLR spotlight talk](https://iclr.cc/virtual/2021/spotlight/3479) — the recorded talk (slides + narration) is genuinely the clearest walkthrough available for this topic.

### D. Uncertainty quantification — for SPS
Goal: understand the coverage-vs-sharpness tradeoff well enough to reason
about interval width choices, since the default band scored so badly above.
- **Conformal prediction** (intro level) — search "conformal prediction
  intro tutorial"; Angelopoulos & Bates' "A Gentle Introduction to Conformal
  Prediction..." (arXiv 2107.07511) is the standard accessible reference.
  You want the core idea (data-driven, distribution-free interval width with
  a coverage guarantee), not the full theory.
- **MC Dropout / deep ensembles** as a simpler alternative uncertainty
  source — Gal & Ghahramani "Dropout as a Bayesian Approximation" (arXiv
  1506.02142) abstract is enough for the concept.
- Once you've read one of these, go back to `scoring.py::aggregate_sps` and
  work out by hand: for a fixed accuracy level, what interval width
  maximizes the SPS reward `(1 - pm) * exp(-(upper-lower)/sigma_global)`? That
  derivation is small and worth doing yourself.

**Visual resources:**
- [MAPIE regression/time-series tutorials](https://mapie.readthedocs.io/en/v0.8.1/examples_regression/4-tutorials/plot_ts-tutorial.html) — official docs with runnable coverage-vs-width plots; the closest thing to "interactive" here since you can execute it yourself.
- [Conformal Prediction Theory Explained — Artem Ryasik](https://medium.com/low-code-for-advanced-data-science/conformal-prediction-theory-explained-14a35226df80) (Medium) — clear stepwise diagrams building intuition before diving into MAPIE.
- [A Tutorial on Conformal Prediction — Shafer & Vovk](https://arxiv.org/pdf/0706.3188) — canonical reference once the visual intros click; not visual, but the precise version to fall back on.

### E. The engineering contract itself
Goal: internalize this well enough that you never violate it by accident.
- Re-read `docs/interface.md` end to end, slowly, a second time now that you
  have the domain context above.
- Read `scoring.py` in full yourself (I've already shown you the contents
  above) — trace `rel_l2_per_sample`, `kinetic_energy`,
  `mvpe_rel_l2_per_sample`, and `aggregate_sps` line by line against the
  formulas in `docs/metrics.md` until they visibly match.
- Read `local_eval.py`'s streaming loop (`build_stream` +  the `for step in
  stream` block) to see exactly how trajectory boundaries, prev-target
  caching, and timing are implemented — this *is* effectively the official
  ingestion loop, mirrored.

---

## 4. Suggested reading order

1. PIV + Kármán vortex street + TKE (short, builds intuition fast)
2. FNO / CNO / Transolver abstracts (know your three tools)
3. TENT → CoTTA-style continual TTA (the heart of the problem)
4. Conformal prediction intro (for SPS)
5. Re-read `docs/interface.md`, `scoring.py`, `local_eval.py` cold, now with
   all of the above as background — things that looked like arbitrary
   formulas the first time should now read as physically/statistically
   motivated.

---

## 5. What "try a baseline submission" required, concretely

Recap of what was actually needed to get *any* submission running (done
above, kept here as a reference checklist for next time):

- [x] Python env with `torch`, `numpy`, `h5py`, `einops`, `pyyaml` (all
      already present — no install needed on this machine).
- [x] `cp submission_template.py submission.py` (the kit ships a template,
      not a ready file — naming matters, `local_eval.py` looks for
      `submission.py` specifically).
- [x] `python3 local_eval.py --submission .` — runs the untrained
      `TinyForecaster` through the full streaming/timing/scoring loop on
      bundled synthetic data. This validates *plumbing* (shapes, timing,
      info dict contract), not real accuracy.
- [x] Same for `agentic_demo/` via `--submission agentic_demo` — validates
      the controller-based approach also satisfies the contract.
- [ ] **Not done yet, deliberately**: downloading a real baseline checkpoint
      (`sim_real_cno.pth` etc. from the competition Google Drive) and wiring
      it through `load_baseline.py` + `ReferenceTTTModel` to get real
      accuracy numbers instead of an untrained placeholder net. That's the
      natural next hands-on milestone once more of the study above is done.
