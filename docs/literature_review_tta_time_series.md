# Test-Time Adaptation for Time Series, Not Images

**A focused companion to `literature_review_test_time_adaptation.md`, covering the research thread that actually studies streaming, sequential prediction under delayed supervision — RealPDE's exact shape.**
**Last updated:** 14 September 2026

---

## Why this document exists

The earlier review (`literature_review_test_time_adaptation.md`) is built almost entirely on image-classification TTA: TENT, MEMO, CoTTA, EATA, RoTTA, SAR. That's not a mistake — those papers *are* where the field's vocabulary and core principles (small parameter subsets, anchoring, don't-update-blindly) were established. But every one of them adapts a model that outputs one label per static image. RealPDE outputs a 20-frame spatiotemporal velocity movie, streamed continuously, with the true answer to *yesterday's* question arriving today.

There is now a real, separate research thread that studies exactly that shape of problem: **test-time adaptation for time series forecasting (TSF-TTA)**, plus an adjacent, longer-running thread on **online/continual learning for streaming forecasts** that shares the same delayed-label structure without always using the word "TTA." Both speak more directly to RealPDE than anything in the image-TTA canon. This document is that thread.

---

## The structural argument for why this is a different problem

[STEPS (2026)](https://arxiv.org/pdf/2605.08005) makes the case explicitly, framing forecasting TTA as its own setting rather than a regression footnote to image TTA:

- the adaptation signal is **short** (you get one revealed window, not a large unlabeled batch),
- it's **temporally correlated** (this window and the next aren't independent samples — they're the same physical process one step later),
- and it's **potentially noisy** (a single measurement, not an average over many).

That combination produces three named failure modes: **weak identifiability** (a short signal underdetermines what actually needs correcting), **error accumulation** (a wrong correction compounds because it's applied again next step, to a correlated input), and **unstable long-horizon correction** (correcting frame 1 of a 20-frame window doesn't tell you what to do to frame 20). All three map directly onto why `docs/AGENT_HANDOFF.md` and the original review both worry about naive per-step SGD compounding over a long real trajectory — this is the same worry, now with a name and a dedicated literature.

STEPS's own answer — reformulating the problem as a boundary-value problem on a temporal manifold, using the revealed prefix error as a boundary condition to solve for the unknown future error field — is more machinery than a v1 needs, but the framing is worth sitting with: it treats "what's the current error, given only the previous one" as a genuine inference problem, not just a threshold to compare against.

---

## Where RealPDE actually sits on the ground-truth spectrum

TSF-TTA papers differ enormously in *how much* of the target they assume is available, and this axis matters more here than in image TTA (which is stuck at "none"):

```text
no ground truth ever          →  classic image TTA (TENT, CoTTA, EATA...)
partially-observed target     →  TAFAS: POGT (partially-observed ground truth)
fully revealed ("matured")    →  RealPDE, and the protocol argued for by
target, but delayed by one    ->  "Towards Principled TTA for TSF"
step
```

[TAFAS (Kim et al., AAAI 2025)](https://arxiv.org/abs/2501.04970) is built around **partially-observed ground truth (POGT)**: in many real deployments you don't wait for the whole forecast horizon to resolve before adapting — you use whatever prefix of the true continuation has become available so far, scheduled by a periodicity-aware component that decides how much POGT to actually use before it's stale. Its input/output **gated calibration modules** adapt small correction layers rather than the backbone, explicitly to "preserve the core semantic information learned during pre-training."

[Towards Principled Test-Time Adaptation for Time Series Forecasting (2026)](https://arxiv.org/abs/2605.17250) then pushes back on the field TAFAS sits in: it diagnoses that existing TSF-TTA methods use revealed targets "in heterogeneous ways" with no unified formulation, and proposes an adaptation protocol built on **matured ground truth** — only adapting once a target is fully, confirmedly revealed, not partially. That is exactly RealPDE's contract (`prev_target_norm` is the complete, genuine previous target, never a partial peek). Worth knowing: what feels like the "obvious, given" setup for this competition is, in the current literature, the cleaner end of a spectrum other papers are still arguing toward. Their own method, **Frequency-Aware Calibration (FAC)**, parameterizes prediction corrections directly in frequency space rather than the time domain — on the reasoning that naive time-domain adapters make "limited and weakly structured" corrections.

**Lesson for RealPDE**: our controller's `err = rel_l2(prev_pred, prev_target_norm)` signal is already using matured ground truth correctly, per the newest guidance in this literature. That's not an accident to second-guess — it's the thing to keep doing.

---

## The dominant design pattern: calibrate, don't retrain

Once you look at TSF-TTA specifically instead of image TTA, one pattern recurs everywhere: **a small calibration module on top of a frozen backbone, not full-model fine-tuning.**

- TAFAS: gated calibration modules on input and output, backbone frozen.
- [PETSA (ICML 2025)](https://arxiv.org/abs/2506.23424): low-rank adapters + dynamic gating, with a loss combining a robust term, a frequency-domain term (preserve periodicity), and a patch-wise structural term. It beats TAFAS on accuracy (127 vs 88 best-MSE wins across benchmarks) while using **up to 33× fewer parameters** at longer windows — the explicit finding is that parameter-efficient calibration is not a compromise, it's often *better*, because it's less prone to overfitting a single noisy revealed window.
- [AdaNODEs (2026)](https://arxiv.org/abs/2601.12893): a source-free TTA method using Neural ODEs to model the continuous-time dynamics of the shift itself, rather than discretely re-fitting weights.

**Lesson for RealPDE**: our v1 controller already follows this pattern structurally — `recalibrate` (additive output correction, no gradient) and `update_adapter` restricted to `adapt_scope: bn` (batch-norm-only, not `all`) are both calibration-style, not full-model fine-tuning. PETSA's result is a concrete argument for staying there rather than expanding `adapt_scope` to `all` in search of more accuracy: this literature's newest, best-performing method is moving *toward* smaller and more structured corrections, not away from them. If anything, it's worth trying a **calibration-only ablation** — drop `update_adapter` from the action space entirely and see whether `recalibrate` alone (tuned to look more like PETSA's gated low-rank correction) closes most of the gap without the anchor-loss machinery at all.

---

## Deciding *when* to adapt, and how much

The image-TTA literature already taught "don't update on everything" (that's the whole reason CoTTA/EATA/SAR exist). The TSF-TTA literature has more specific machinery for it:

- [Detect-then-Adapt / D3A (2024)](https://arxiv.org/abs/2403.14949): a two-phase strategy — detect concept drift first, then adapt aggressively *only once drift is confirmed*, using Gaussian-noise augmentation to close the resulting train/test distribution gap. The explicit argument is that adapting on every step causes "continuous performance regression" as small, unnecessary updates accumulate.
- [DynaTTA, from "Shift-Aware Test-Time Adaptation and Benchmarking for Time-Series Forecasting" (ICML 2025)](https://openreview.net/forum?id=a399SmgWGl): tracks prediction error *and* embedding drift together, and uses them to set a **continuous, severity-scaled adaptation rate** plus shift-conditioned gating — rather than a discrete decision between a few fixed actions. It ships alongside **TTFBench**, a benchmark built specifically because standard TSF datasets have too little real distribution shift in their test splits to evaluate TTA methods meaningfully.
- [RG-TTA (2026)](https://arxiv.org/html/2603.27814): the most structurally interesting one for us. It maintains a memory of previously-seen **regimes** and scores each incoming window's similarity to them (via an ensemble of KS-statistic, Wasserstein-1, feature-distance, and variance-ratio metrics). A meta-controller scales the learning rate by that similarity — aggressive on genuinely novel regimes, conservative on familiar ones — and can **load a stored specialist checkpoint** for a previously-seen regime instead of adapting from scratch, gated so a specialist only loads if it would demonstrably beat the current model.

**Lesson for RealPDE**: our v1's `err_low`/`err_high` split is a discrete, 2-threshold version of exactly this idea, and it's a reasonable v1. But RG-TTA's regime-memory framing is unusually well-matched to our data specifically — RealPDE's real trajectories *are* already regime-structured, by `(Re, AoA)`. A trajectory at `Re=3750` and one at `Re=26700` are legitimately different flow regimes (attached vs. separated boundary layer, different shedding character), not noise around one distribution. A v2 direction worth taking seriously: instead of resetting to one frozen checkpoint every trajectory boundary and re-learning the same corrections from scratch each time, keep a small memory of per-regime calibration states (keyed by cheap early-window statistics) and warm-start from the closest one. DynaTTA's continuous rate-scaling is also a cleaner idea than our hard `err_low`/`err_high` cutoffs, if the discrete rule ever shows threshold-boundary artifacts in practice.

---

## A cautionary result that rhymes with our own experiment this session

[Test-Time Adaptation for Non-stationary Time Series: From Synthetic Regime Shifts to Financial Markets (2026)](https://arxiv.org/abs/2602.00073) freezes the backbone and adapts only normalization affine parameters, with a drift penalty and an uncertainty-triggered fallback to keep updates conservative. Its headline finding: **norm-only adaptation reliably helps on synthetic drift, but on real financial markets, more aggressive adaptation can actively hurt** — their conclusion is that a plain batch-norm statistics update is "a robust default" for real data, ahead of anything fancier.

This is worth flagging because it's an independent confirmation, from a totally different domain, of the exact caution this project already needs: **a design validated on clean/synthetic data does not automatically transfer its aggressiveness to real, noisier data.** We saw a version of this ourselves this session — the same v1 controller and checkpoint scored very differently on `sps_score` between real and sim trajectories (21.7 vs 0.15), because constants calibrated for real data's physical scale don't hold on sim's. The financial-markets paper's version of the same lesson is sharper: don't just confirm a hyperparameter (`anchor_lambda`, `err_low`/`err_high`, `sps_k`) looks reasonable on one domain and assume it transfers — validate specifically on real held-out trajectories, which is exactly what `docs/AGENT_HANDOFF.md`'s recommended next steps already call for.

---

## The adjacent thread: online learning for streaming forecasts

These papers rarely use the word "TTA," but they study the identical problem shape — a forecaster that keeps seeing new windows and delayed feedback, forever, not just once at deployment:

- [FSNet (Pham et al., ICLR 2023)](https://arxiv.org/abs/2202.11672) — "Learning Fast and Slow," inspired by Complementary Learning Systems theory. A per-layer **fast adapter** handles abrupt local changes; a separate **associative memory** remembers, updates, and recalls *recurring* patterns, so the model doesn't have to re-learn a pattern it has already seen before. If RealPDE trajectories ever revisit similar flow regimes (plausible, given the Re/AoA grid structure), an associative-memory-style component is a more principled way to reuse that than treating every trajectory as fully independent.
- [OneNet (NeurIPS 2023)](https://arxiv.org/abs/2309.12659) — maintains two complementary forecasters (one modeling temporal dependency, one cross-variate dependency) and online-ensembles them with reinforcement-learning-adjusted weights, specifically to react faster to concept drift than either model alone. The "cross-variate" half doesn't map cleanly onto a single velocity field, but the general principle — ensemble a frozen and an adapting predictor, weight them online by which has been more reliable recently — is a plausible, low-risk hedge against a bad `update_adapter` step: blend the adapted prediction with the frozen checkpoint's, weighted by recent relative accuracy, rather than fully committing to whichever the rule picked.

---

## The cheapest lever, named properly: instance normalization

[RevIN (Kim et al., ICLR 2022)](https://openreview.net/forum?id=cGDAkQo1C0p) predates the "TTA for forecasting" label but is arguably its ancestor: normalize the input window by *its own* instance mean and standard deviation, run the model, then de-normalize the output using the same statistics (plus a learned affine transform). It's now close to universal in modern forecasting backbones (PatchTST, iTransformer).

RealPDE's current normalization is the opposite of this: `mean_std_real.pt` is a single, fixed statistic computed once, offline, over the *entire* training release — every window gets normalized by the same global `(mean, std)` regardless of which trajectory or regime it's from. That's necessary for consistency with how the checkpoints were trained, but it also means we've never tried the cheapest form of test-time adaptation that exists: recomputing a *local* correction to normalization per revealed window, the way RevIN does per-instance. This is a smaller, more defensible experiment than either `recalibrate` or `update_adapter` — worth trying before reaching for anything with a gradient.

---

## The neighboring but genuinely different idea from physics

[Serrano et al., Test-time Generalization for Physics through Neural Operator Splitting (2026)](https://arxiv.org/abs/2602.00884) — already cited in the original review, and confirmed here independently — takes a structurally different approach: no gradient updates at all, just **searching over compositions of pretrained operators** to approximate unseen dynamics. It's not the same delayed-label protocol RealPDE uses, but it's the reminder that "test-time adaptation" doesn't have to mean "test-time gradient descent." If gradient-based calibration ever plateaus, composition/search over a small library of frozen adapters (rather than one continuously-updated one) is the structurally different direction to reach for next, not a bigger version of the same thing.

---

## What this changes, concretely

Ranked by how directly each is testable against the existing v1 controller:

1. **Calibration-only ablation.** Run v1 with `update_adapter` removed from the action space (`err_high` effectively infinite, or a policy variant), keeping only `skip_update`/`recalibrate`. PETSA and TAFAS both suggest this might not cost much accuracy — worth knowing whether the anchor-loss machinery is pulling its weight.
2. **Validate thresholds on real data specifically**, not sim, before trusting them — the financial-markets paper's finding, and our own SPS gap this session, are two independent confirmations of the same risk.
3. **Regime-aware warm-starting (v2 direction)**: RG-TTA's per-regime specialist memory maps unusually well onto RealPDE's `(Re, AoA)` structure — a natural next experiment once v1 is validated, not a v1 requirement.
4. **Try per-window instance-normalization correction** (RevIN-style) as a cheaper alternative/complement to `recalibrate`, before assuming a costlier method is needed.
5. **Continuous severity-scaled adaptation rate** (DynaTTA-style) as a replacement for the discrete `err_low`/`err_high` split, if the hard threshold ever shows boundary artifacts in practice.

---

## Reading list

1. [STEPS (2026)](https://arxiv.org/pdf/2605.08005) — why forecasting TTA is a structurally different problem from image TTA.
2. [TAFAS, Kim et al. (AAAI 2025)](https://arxiv.org/abs/2501.04970) — partially-observed ground truth, gated calibration modules.
3. [Towards Principled TTA for TSF (2026)](https://arxiv.org/abs/2605.17250) — matured ground truth protocol, frequency-domain calibration.
4. [PETSA (ICML 2025)](https://arxiv.org/abs/2506.23424) — parameter-efficient calibration beats full-model adaptation.
5. [AdaNODEs (2026)](https://arxiv.org/abs/2601.12893) — Neural-ODE-based source-free TTA.
6. [Detect-then-Adapt / D3A (2024)](https://arxiv.org/abs/2403.14949) — drift detection gates aggressive adaptation.
7. [DynaTTA / Shift-Aware TTA + TTFBench (ICML 2025)](https://openreview.net/forum?id=a399SmgWGl) — continuous severity-scaled adaptation, dedicated shift benchmark.
8. [RG-TTA (2026)](https://arxiv.org/html/2603.27814) — regime memory, similarity-gated specialist checkpoints.
9. [TTA for Non-stationary TS: Synthetic to Financial Markets (2026)](https://arxiv.org/abs/2602.00073) — norm-only adaptation helps on synthetic, can hurt on real.
10. [FSNet, Pham et al. (ICLR 2023)](https://arxiv.org/abs/2202.11672) — fast per-layer adapters + associative memory for recurring patterns.
11. [OneNet (NeurIPS 2023)](https://arxiv.org/abs/2309.12659) — online ensembling of complementary forecasters.
12. [RevIN, Kim et al. (ICLR 2022)](https://openreview.net/forum?id=cGDAkQo1C0p) — the ancestor: per-instance normalize/denormalize.
13. [Serrano et al. (2026)](https://arxiv.org/abs/2602.00884) — operator composition search, the non-gradient alternative.

---

## The one-paragraph takeaway

Image-classification TTA (TENT, CoTTA, and the rest) established the field's vocabulary, but a dedicated time-series-forecasting-TTA literature now exists and speaks far more directly to RealPDE: it agrees that RealPDE's fully-revealed, one-step-delayed target ("matured ground truth") is the clean end of the spectrum other papers are still working toward; it converges, independently of the image-TTA literature, on calibration over full fine-tuning as the dominant winning pattern; it has more developed machinery for deciding *when and how much* to adapt (drift detection, regime memory, continuous severity scaling) than our v1's two fixed thresholds; and it has already produced a real-world cautionary tale — synthetic-validated adaptation aggressiveness not transferring safely to real data — that rhymes exactly with what we saw ourselves comparing v1 on real vs. sim trajectories this session.
