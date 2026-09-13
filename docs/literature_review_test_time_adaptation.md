# Test-Time Adaptation: a Literature Review for Streaming Physical Forecasting

**Prepared:** 13 September 2026  
**Scope:** Test-time adaptation (TTA), test-time training (TTT), continual TTA, and adjacent online uncertainty methods. This is a curated technical review rather than a claim to enumerate every one of the field's hundreds of papers. The broadest recent survey reports more than 400 TTA papers and is the best map for exhaustive searching: [Xiao & Snoek, 2024](https://arxiv.org/abs/2411.03687).

**Why this document is tailored to RealPDE Track 2:** most headline TTA work is *unlabelled image classification*. This competition is materially different: it is a causal, long-lived **regression/forecasting** stream in which the correct output from the preceding window is revealed with a one-step delay. That delayed target is unusually valuable: it permits genuine supervised online learning. Methods that only minimize classification entropy should therefore be treated as ideas about *stability and update control*, not copied literally.

---

## 1. The problem in one mathematical picture

A deployed predictor was trained on source data:

\[
f_{\theta_0}: x \mapsto \hat y.
\]

At deployment, the joint distribution changes from \(P_S(x,y)\) to \(P_T(x,y)\). This can happen through sensor noise, changed geometry, new operating conditions, numerical-vs-real mismatch, or an evolving environment. Ordinary inference freezes \(\theta_0\); TTA uses evidence available at deployment to form \(\theta_t\) and predict better on the target stream.

The generic causal loop is:

```text
receive x_t → choose/update state θ_t using only permitted past information → predict ŷ_t
                                                    ↓
                                  later receive feedback about y_t, if the protocol allows it
```

The central design question is not merely “how can we update?” but:

> **Which signal is trustworthy enough to justify an update, which small part should change, and how do we prevent accumulated damage?**

### The RealPDE specialization

At stream step \(t\), the competition supplies current input \(x_t\) and the *previous* target \(y_{t-1}\). A legal supervised update is therefore

\[
L_{\mathrm{sup},t-1}(\theta) = \ell(f_\theta(x_{t-1}),y_{t-1}),
\qquad
\theta_t \leftarrow \theta_{t-1}-\eta\nabla_\theta L_{\mathrm{sup},t-1}.
\]

It must happen before predicting \(\hat y_t=f_{\theta_t}(x_t)\), and it must never use \(y_t\). This is online learning with delayed labels, not the usual “fully unlabelled TTA” setting.

---

## 2. Terminology that prevents common confusion

| Term | What is available at deployment? | Typical update signal | Relation to this competition |
|---|---|---|---|
| **Domain generalization (DG)** | No target adaptation data during deployment | None; robustness is learned beforehand | The frozen checkpoint is the DG-style baseline. |
| **Unsupervised domain adaptation (UDA)** | Source data and an unlabelled target set | Domain alignment / pseudo-labels | Usually offline and not allowed/practical here. |
| **Source-free DA** | Source model, target set; no source samples | Pseudo-labels, information maximization | Close in privacy constraints, but often assumes repeated passes over a target dataset. [SHOT](https://arxiv.org/abs/2002.08546) is a landmark. |
| **Fully test-time adaptation (FTTA)** | Source model and incoming unlabelled test samples | Entropy, consistency, normalization stats | The classic TENT-style setting. |
| **Test-time training (TTT)** | A model prepared with an auxiliary task plus test input | Self-supervised auxiliary loss | Requires special preparation during original training. |
| **Continual TTA (CTTA)** | A temporally correlated, non-stationary unlabelled stream | As FTTA, plus anti-forgetting mechanisms | Highly relevant conceptually: RealPDE is long-running. |
| **Online supervised adaptation** | Delayed or immediate true labels | Actual task loss | The closest formal match to Track 2. |

Two abbreviations collide in the literature: “TTA” can mean **test-time adaptation** or **test-time augmentation**. This document uses TTA only for adaptation; it spells out augmentation.

---

## 3. Historical development and foundational methods

### 3.1 Before modern TTA: adapting predictions or statistics

The oldest practical family does not learn all weights. It recalculates Batch Normalization (BN) statistics on target data, or blends source and target statistics. If an activation is normalized as

\[
z'=(z-\mu)/\sqrt{\sigma^2+\epsilon},
\]

then changed sensor/style statistics can be partly repaired by estimating \(\mu,\sigma^2\) from target activations. This is cheap and often a strong safety baseline, but it depends on suitable normalization layers and enough representative samples. It can become unstable with batch size one, correlated frames, or quickly changing regimes.

### 3.2 Test-Time Training (Sun et al., 2020)

[Sun et al.](https://arxiv.org/abs/1909.13231) introduced the influential idea of adding a self-supervised task during source training (their example predicts image rotation). At test time, the model optimizes this auxiliary task on the current unlabelled input before making the main prediction.

**Contribution:** it established that a test input can itself provide a learning signal.  
**Weakness:** success requires an auxiliary task that is correlated with the real task; improving rotation prediction does not guarantee better physical forecasting.  
**RealPDE translation:** temporal consistency, reconstruction, masked-field prediction, or a physics residual could be auxiliary signals, but the delayed true target is more direct and normally preferable.

### 3.3 TENT (Wang et al., ICLR 2021)

[TENT](https://openreview.net/forum?id=uXl3bZLkr3c) made fully test-time adaptation simple and popular. It minimizes predictive entropy

\[
H(p_\theta(y\mid x))=-\sum_c p_c\log p_c
\]

and typically updates only the affine scale/shift parameters of normalization layers. Low entropy means a more confident classifier.

**Why it mattered:** no source data, no target labels, no source-training modification, and a small update set.  
**Failure mode:** confidence is not correctness. In classification, entropy minimization can reinforce a wrong prediction; in regression such as velocity forecasting, there is no natural softmax class entropy, so the literal objective is inappropriate.  
**Transferable lesson:** adapt a restricted, cheap parameter subset and monitor whether updates are trustworthy.

### 3.4 Single-sample and augmentation-based adaptation: MEMO (2022)

[MEMO](https://arxiv.org/abs/2110.09506) handles the case where one test point is available. It creates several valid augmentations \(a_i(x)\), averages the predictive distributions, and minimizes the entropy of that marginal prediction:

\[
\bar p=\frac{1}{K}\sum_{i=1}^{K}p_\theta(y\mid a_i(x)), \qquad L=H(\bar p).
\]

It encourages invariance across augmentations and confidence. It is general for probabilistic models but costs many forward passes. For fluid fields, arbitrary image transformations are physically dangerous: rotating, flipping, or cropping can change boundary conditions and flow direction. Only transformations that preserve the problem's physics should be considered.

### 3.5 Optimization-free adaptation: T3A and LAME (2021–22)

[T3A](https://papers.nips.cc/paper_files/paper/2021/hash/1415fe9fea0fa1e45dddcff5682239a0-Abstract.html) keeps the feature extractor fixed and adjusts class prototypes/templates using confident target features. [LAME](https://arxiv.org/abs/2201.05718) adapts predictions by graph-based Laplacian-adjusted maximum likelihood rather than gradient-updating model parameters.

**Importance:** these methods demonstrate a useful principle for strict runtime budgets: store/update a small state outside the main model rather than backpropagating through a large network.  
**RealPDE analogue:** an output bias/scale correction, low-rank adapter, residual corrector, or small temporal calibration state can be safer and faster than modifying the full neural operator.

---

## 4. The main families of current TTA techniques

The field can be organized by *what changes at test time*. The categories below synthesize the taxonomy in [Xiao & Snoek's survey](https://arxiv.org/abs/2411.03687), with an emphasis on mechanisms rather than a list of method names.

### A. Normalization adaptation

**State changed:** BN running mean/variance and sometimes affine parameters \(\gamma,\beta\).  
**Signal:** target activation statistics, optionally entropy.  
**Cost:** very low to low.  
**Representative:** AdaBN-style statistical replacement, TENT, robust BN schemes in [RoTTA](https://arxiv.org/abs/2303.13899).

**Strengths:** easy, fast, little memory, preserves most pretrained weights.  
**Risks:** a tiny/correlated batch is not a population; running statistics can chase a changing stream. CNO/FNO/transformer architectures may not expose conventional BN parameters, so applicability is architecture-dependent.

### B. Entropy minimization and information maximization

**State changed:** a subset or all model parameters.  
**Signal:** make predictions confident; some variants also encourage diverse predictions across a batch to avoid one-class collapse.  
**Cost:** at least one backward pass per update.  
**Representative:** TENT; source-free [SHOT](https://arxiv.org/abs/2002.08546); robust/stable [SAR](https://arxiv.org/abs/2302.12400).

**Strengths:** label-free and broadly applicable to probabilistic classifiers.  
**Risks:** minimized entropy can yield confidently wrong predictions, class collapse, and unsafe parameter drift. For continuous vector fields, one must invent a replacement signal; direct delayed MSE is stronger whenever a past target is available.

### C. Pseudo-label self-training and teacher–student methods

**State changed:** student model; teacher may be an exponential moving average (EMA).  
**Signal:** the model/teacher's confident predictions become temporary labels; augmentation consistency is often added.  
**Representative:** [CoTTA](https://arxiv.org/abs/2203.13591), RoTTA.

CoTTA averages predictions over augmentations, maintains an EMA teacher, and randomly restores a small fraction of weights to their source values. The restoration is a direct answer to long-run forgetting.

**Strengths:** can exploit a stream and smooth noisy pseudo-labels.  
**Risks:** an early wrong teacher can teach the student the wrong thing; multiple augmentations/teacher copies add latency and memory. In RealPDE, real delayed labels remove much of the reason to trust pseudo-labels.

### D. Consistency, reconstruction, and masked/self-supervised objectives

**State changed:** selected parameters or a small test-time head.  
**Signal:** predictions should agree under valid transformations, reconstruct masked input portions, or satisfy an auxiliary task.  
**Representative:** original TTT; MEMO; many masked-image-modeling variants.

**Strengths:** usable without labels and can work at batch size one.  
**Risks:** proxy-task improvement may be unrelated to prediction improvement. Every augmentation has to respect domain semantics—an especially strong constraint for PDE fields.

### E. Feature/distribution alignment

**State changed:** feature extractor, normalizer, or an input transformation.  
**Signal:** align target feature moments, covariances, source prototypes, or stored source statistics.  
**Representative:** [CAFe](https://arxiv.org/abs/2204.13263) aligns covariance-aware feature statistics.

**Strengths:** directly targets representation shift; precomputed source summaries can be stored without source examples.  
**Risks:** alignment assumes the target should resemble source features; it can be wrong under semantic/concept shift. Covariance estimates are noisy in low-batch settings.

### F. Memory banks and replay-like mechanisms

**State changed:** usually a small memory plus model parameters.  
**Signal:** a curated buffer of recent, diverse, confident, or class-balanced test examples supports more stable updates.  
**Representative:** RoTTA; [ResiTTA](https://arxiv.org/abs/2401.14619).

**Strengths:** counters the fact that consecutive stream samples are correlated and a single mini-batch is unrepresentative.  
**Risks:** stale memory harms responsiveness; memory selection itself can amplify biased/confident mistakes; strict competition limits make large buffers unattractive.

### G. Regularized, selective, and safe parameter updates

**State changed:** restricted parameters, with a penalty or recovery mechanism.  
**Signal:** standard TTA loss plus a stability constraint.  
**Representative:** [EATA](https://arxiv.org/abs/2204.02610), SAR, CoTTA.

EATA selects reliable/non-redundant samples before entropy adaptation and uses a Fisher-information regularizer to protect important source parameters. SAR filters unreliable samples, uses sharpness-aware optimization, and has an episodic recovery mechanism when instability is detected.

**Key lesson for this competition:** *not every observed target window deserves an update of the same size*. This family is the most directly useful conceptual template for an update controller.

### H. Parameter-efficient modules: adapters, prompts, and low-rank updates

**State changed:** a small adapter, a prompt, normalization affine values, or low-rank parameters—not the backbone.  
**Signal:** entropy, consistency, pseudo-labels, or supervision.  
**Representative:** [Test-time Prompt Tuning (TPT)](https://arxiv.org/abs/2209.07511) for CLIP optimizes prompt context using entropy over augmentations.

**Strengths:** small state, lower forgetting risk, easier reset, and low optimizer memory.  
**Risks:** capacity may be inadequate for a large physical shift; prompt-specific work does not transfer directly to neural operators.  
**RealPDE relevance:** a residual adapter added around the pretrained forecast is an attractive first design because it is fast, resettable, and limits damage to the base model.

### I. Test-time inference, ensembling, and sampling rather than weight updates

**State changed:** no weights; computation or prediction aggregation changes.  
**Signal:** multiple augmentations, model ensemble, Bayesian posterior approximation, or a learned cache.  
**Examples:** test-time augmentation, MC dropout, temporal ensembling, and the test-time operator-search approach in [Serrano et al., 2026](https://arxiv.org/abs/2602.00884).

**Strengths:** avoids catastrophic forgetting.  
**Risks:** compute cost can dominate; it may not correct a persistent bias.  
**RealPDE relevance:** lightweight ensembles can estimate uncertainty for SPS, but repeated full neural-operator passes can hurt the time score.

---

## 5. Continual TTA: why long streams are harder

Classic papers often evaluate one static corruption distribution. In deployment, distributions can change gradually, return to old regimes, or mix within a stream. The risks are:

1. **Error accumulation:** incorrect pseudo-labels or noisy updates create worse future labels/signals.
2. **Catastrophic forgetting:** adaptation improves the recent target domain but damages original/general behavior.
3. **Model collapse:** entropy objectives can converge to one class or a degenerate representation.
4. **Correlated observations:** adjacent video/PIV frames are not independent; treating them as a diverse batch exaggerates confidence.
5. **Label/distribution imbalance:** some regimes dominate a memory buffer or update sequence.
6. **Hyperparameter leakage:** picking learning rate/thresholds using hidden target labels is invalid; even in research, online tuning is difficult.

CoTTA's EMA teacher, augmentation averaging, and stochastic source-weight restoration are canonical continual defenses. EATA contributes reliable sample selection and Fisher protection. RoTTA adds time-aware memory and robust normalization. SAR targets unstable/noisy gradients and recovery. These differ in details but share one theme: **constrain and audit adaptation rather than performing unconstrained SGD indefinitely.**

[TTAB: On Pitfalls of Test-Time Adaptation](https://arxiv.org/abs/2306.03536) is essential reading before accepting benchmark claims. Across methods and shifts, it found that hyperparameter/model selection is difficult online, method quality strongly depends on the starting model, and no method reliably handles every common shift.

---

## 6. Regression, forecasting, and physical systems: what changes

### 6.1 Entropy is not the native objective

Classification methods output a categorical distribution \(p(y\mid x)\), so entropy is straightforward. A flow forecaster outputs a continuous tensor \(\hat y\in\mathbb{R}^{T\times H\times W\times C}\). If a model does not output a calibrated probability distribution, “minimize entropy” needs an artificial construction and can simply shrink predicted variation—bad for turbulence metrics.

For RealPDE, once \(y_{t-1}\) is available, a direct loss is valid:

\[
L_{\mathrm{MSE}}=\frac{1}{N}\sum_i(\hat y_i-y_i)^2.
\]

Better task-aware variants may combine velocity error with terms aligned to the scoreboard (for example, temporal variance/TKE or probe-profile terms), but any such loss needs careful validation to avoid trading one metric against another.

### 6.2 Causality and delayed supervision

Forecasting is not a generic unordered test set. The update at step \(t\) can use only previous windows:

\[
\theta_t = U(\theta_{t-1},x_{t-1},y_{t-1}),\quad \hat y_t=f_{\theta_t}(x_t).
\]

This is a stronger, cleaner signal than pseudo-labels, but one window is still noisy and may be non-representative. It motivates delayed-label online learning ideas: recency-weighted errors, change-point/update triggers, trust regions around the initial model, and small adaptive heads.

### 6.3 Physics-based auxiliary signals

For a nominally incompressible two-dimensional velocity field, a possible diagnostic is the divergence residual:

\[
\nabla\cdot\mathbf{u}=\partial u/\partial x+\partial v/\partial y\approx0.
\]

One might penalize this on predictions or use it as a quality gate. However, this is not automatically a good loss: PIV noise, boundaries, masking around the airfoil, grid spacing, and the 2D measurement of a potentially 3D flow all affect the residual. Treat it as a carefully tested regularizer/diagnostic, never as proof that a prediction is correct.

### 6.4 Test-time generalization for neural operators

This is a small but emerging niche. [Serrano et al. (2026)](https://arxiv.org/abs/2602.00884) search over compositions of pretrained neural operators at test time to represent unseen dynamics without changing model weights. This is not the same protocol as RealPDE's delayed labels, but it is relevant evidence that **test-time computation need not mean gradient updates**. Its cost may be too high for the competition's per-step timing constraint.

---

## 7. Uncertainty and online calibration

TTA and uncertainty are coupled: adaptation can make a model more confident without making it more correct. RealPDE separately rewards prediction intervals through SPS, so uncertainty must be tracked rather than treated as an afterthought.

### Useful approaches

| Method | Basic idea | Cost | Caveat for a stream |
|---|---|---:|---|
| Residual-scale tracker | Estimate recent error quantiles/scales from delayed labels and use them to size intervals | Very low | Must adapt to shift; a single global width ignores spatial heterogeneity. |
| Ensemble / MC dropout | Prediction disagreement estimates epistemic uncertainty | High | Requires multiple forward passes and may harm time score. |
| Heteroscedastic head | Model predicts mean and variance, trained with likelihood | Moderate | Needs suitable source training and calibrated variance. |
| Conformal prediction | Calibrate intervals with past nonconformity scores | Low–moderate | Classical coverage needs exchangeability, violated by evolving correlated flows. |
| Online/adaptive conformal | Update conformal threshold sequentially | Low | Coverage becomes adaptive/empirical rather than a simple finite-sample guarantee under drift. |

[Angelopoulos & Bates](https://arxiv.org/abs/2009.14193) is the standard accessible introduction to conformal prediction. The important conceptual distinction is **coverage versus sharpness**: an interval covering everything is safe but uninformative. RealPDE SPS explicitly rewards both coverage and narrow width.

For this task, a reasonable first research baseline is to collect per-element or grouped residual magnitudes from *previous revealed windows*, calculate a robust recent quantile, and apply a controlled interval around the current prediction. It must be tested chronologically and reset at trajectory boundaries.

---

## 8. What literature suggests for a RealPDE strategy

### High-value ideas to test first

1. **Frozen pretrained base + small supervised residual adapter.** Keep \(f_{\theta_0}\) fixed and learn \(g_\phi\) where \(\hat y=f_{\theta_0}(x)+g_\phi(x)\). Reset \(\phi\) per trajectory. This imports the parameter-efficient/safe-update principle while exploiting valid delayed targets.
2. **Selective updates.** Gate updates by recent loss, gradient norm, change in input statistics, or a held-out portion of the prior window. EATA/SAR motivate refusing unreliable updates; use actual previous-target error rather than classifier entropy.
3. **Anchored updates.** Penalize change from initialization, \(\lambda\|\phi-\phi_0\|^2\), or periodically interpolate back toward the checkpoint. This is the direct analogue of anti-forgetting restoration.
4. **Update schedule as a control policy.** Compare no update, every step, every \(k\) steps, and evidence-triggered updates. The Time score makes an “update only when worthwhile” policy especially relevant.
5. **Chronological uncertainty calibration.** Estimate interval width from past residuals only, with a small recent buffer and a trajectory reset. This addresses SPS without pretending that default percentage bands are calibrated.

### Ideas that are tempting but need extra skepticism

- Directly applying TENT entropy to velocity tensors: no clear probabilistic interpretation and may suppress true turbulence.
- Strong visual augmentations: may break airfoil/wake physics.
- Full-network SGD every step: expensive and prone to drift.
- Large memory banks/teachers/ensembles: potentially useful but risky under a 10-minute total runtime and per-step time score.
- Optimizing a proxy physics residual alone: it can produce physically smooth but inaccurate fields.
- Choosing an approach from only two synthetic local trajectories: the kit explicitly says those numbers are not leaderboard-comparable.

### A disciplined experiment order

```text
0. Frozen real-finetuned CNO/Transolver baseline; record all five subscores.
1. One tiny delayed-supervised adapter update; sweep learning rate and update frequency.
2. Add anchoring / gradient clipping / loss-based skip rule.
3. Add a lightweight residual-based interval calibrator for SPS.
4. Evaluate long trajectories chronologically, including shifts and resets.
5. Only then investigate expensive teachers, physics terms, or full-model adaptation.
```

Do not select a method by mean MSE alone. Record time, per-step errors over time, TKE, wake probes, interval coverage/width, and behavior immediately after resets. A method that improves early windows but drifts later is exactly the kind of failure continual-TTA research warns about.

---

## 9. Recommended reading path

1. [Sun et al., Test-Time Training (2020)](https://arxiv.org/abs/1909.13231) — origin of self-supervised test-time learning.
2. [Wang et al., TENT (2021)](https://openreview.net/forum?id=uXl3bZLkr3c) — minimal fully unlabelled adaptation baseline.
3. [Zhang, Levine & Finn, MEMO (2022)](https://arxiv.org/abs/2110.09506) — single-sample adaptation and augmentation consistency.
4. [Wang et al., CoTTA (2022)](https://arxiv.org/abs/2203.13591) — continual adaptation and anti-forgetting restoration.
5. [Niu et al., EATA (2022)](https://arxiv.org/abs/2204.02610) — sample selection plus Fisher regularization.
6. [Niu et al., SAR (2023)](https://arxiv.org/abs/2302.12400) — stability under noisy/mixed online shifts.
7. [Zhao et al., TTAB pitfalls (2023)](https://arxiv.org/abs/2306.03536) — essential counterweight to optimistic claims.
8. [Xiao & Snoek survey (2024)](https://arxiv.org/abs/2411.03687) — broad map of model, inference, normalization, sample, and prompt adaptation.
9. [Angelopoulos & Bates, conformal prediction (2021)](https://arxiv.org/abs/2009.14193) — uncertainty intervals and coverage.
10. [Serrano et al., test-time neural-operator generalization (2026)](https://arxiv.org/abs/2602.00884) — emerging physics/operator-specific perspective.

---

## 10. Bottom line

The modern TTA literature has converged on a cautious conclusion: adaptation is valuable, but indiscriminate adaptation is fragile. The most robust ideas are restricted state changes, trustworthy update selection, explicit anti-forgetting mechanisms, and honest uncertainty calibration.

For RealPDE, the decisive advantage is the delayed ground truth. Use it. Start with small, anchored **supervised** online updates and interval calibration. Borrow the continual-TTA literature's defenses against drift—not its classification entropy objective verbatim.
