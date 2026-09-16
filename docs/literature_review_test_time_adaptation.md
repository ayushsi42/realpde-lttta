# Test-Time Adaptation, Told as a Story

**A beginner-friendly guide to the ideas behind TTA, and why they matter for RealPDE Track 2.**
**Last updated:** 13 September 2026

This is not meant to be a catalogue of acronyms. It is a story about a sensible question:

> A model was trained in one world. What should it do when it is deployed in a slightly different world and starts making mistakes?

The short answer is **test-time adaptation (TTA)**: let the model adjust while it is being used. The difficult part is deciding how to adjust without labels, without damaging what the model already knows, and without spending too much time.

The field is large—one recent survey maps more than 400 papers: [Xiao & Snoek, 2024](https://arxiv.org/abs/2411.03687). This guide focuses on the major ideas you will repeatedly encounter and translates every one into the RealPDE setting.

**A note on scope:** almost everything below comes from image classification, because that is where TTA's vocabulary was established. A separate, more recently active thread studies test-time adaptation for *time series forecasting* specifically — closer in shape to RealPDE's streaming, delayed-target protocol. See [`literature_review_tta_time_series.md`](literature_review_tta_time_series.md) for that thread; read this one first for the foundational vocabulary, then that one for the closer analogues.

---

## Before the story: the one thing RealPDE gives us that most TTA papers do not

Most famous TTA papers are about an image classifier in a new environment. Imagine a system trained to recognise dogs and cats on clean photos, then deployed on blurry phone photos. It sees the new photos, but nobody tells it whether “dog” or “cat” was correct.

RealPDE is kinder than that—but only a little kinder.

At window $t$, you receive 20 observed flow frames, $x_t$, and must predict the next 20 frames, $hat y_t$. You cannot see the correct future, $y_t$, yet. At the following call, the true previous target $y_{t-1}$ is revealed.

```text
time t:       receive x_t       → make prediction ŷ_t
time t + 1:   receive x_{t+1}   → learn whether ŷ_t was good, using y_t
```

That means we are not forced to guess whether an update is good. We can calculate a real error on the previous window:

$$
L_{t-1} = \operatorname{MSE}(f_\theta(x_{t-1}), y_{t-1}).
$$

This is called **delayed supervised online learning**. It is the closest description of the contest. Classic unlabelled TTA is still valuable, but mainly because it teaches us how not to drift, collapse, or overreact.

---

## Act 1: The reliable student meets the real world

Suppose you train a student using thousands of examples of air flowing past an airfoil in a simulator. The student becomes good at predicting the next part of the flow. Then you take them to a wind tunnel.

The real experiment differs in small, important ways:

- the PIV camera measurement is noisy;
- the exact airfoil setup or operating condition differs;
- real turbulence has details that a simulator did not perfectly reproduce;
- the new trajectory can gradually change over time.

The model has not suddenly become unintelligent. It has encountered **distribution shift**: the statistical world at deployment differs from the one it learned from.

In symbols, source training data comes from $P_S(x,y)$ and deployment data comes from $P_T(x,y)$. Usually, $P_S \ne P_T$.

The naïve response is to freeze the model:

```text
current flow window → fixed model → future-flow prediction
```

This is stable and fast. But if the model has a persistent bias on this wind-tunnel run, it never corrects itself.

The opposite response is also naïve:

```text
every new observation → change every model weight as much as possible
```

This can work for a moment, then make the model worse. The rest of this story is about finding the middle ground.

---

## Act 2: “Can the input teach the model something without an answer?”

The first modern answer was **Test-Time Training (TTT)** by [Sun et al.](https://arxiv.org/abs/1909.13231).

### The idea

During normal training, give the model two jobs:

1. the real job, such as classifying an image;
2. a small self-supervised side job, such as identifying whether that image was rotated.

At deployment, even without the correct class label, the model still knows whether *we* rotated the image. So it can practise the side job on the new image before making its main prediction.

```text
new, unlabelled image
        ↓
known artificial transformation, such as rotation
        ↓
update model on “which rotation did I apply?”
        ↓
make the real classification prediction
```

The hope is that becoming better at the side job on this new kind of image improves the internal representation needed for the main job too.

### Why it was important

It changed the question from “we have no labels, so we cannot learn” to “can we create a useful learning signal from the input itself?” This was a major conceptual jump.

### The catch

The side job has to genuinely help the main job. A model can become excellent at recognising rotations while becoming no better at recognising animals. This is the **proxy-objective problem**: you optimised what was measurable, not necessarily what mattered.

### Lesson for RealPDE

We could invent side jobs: reconstruct masked parts of a flow field, enforce consistency between nearby time windows, or penalize unphysical divergence. But RealPDE later gives us the actual answer for the previous window. That direct error signal is normally more valuable than a proxy. Use self-supervision as a supplement or safety signal, not as the main source of truth.

---

## Act 3: “A confused classifier should become less confused” — TENT

The most influential fully unlabelled TTA method is [TENT](https://openreview.net/forum?id=uXl3bZLkr3c), from Wang et al. (ICLR 2021).

### The intuition

A classifier outputs probabilities. For a photo it might say:

```text
dog: 0.97, cat: 0.02, bird: 0.01     confident
dog: 0.37, cat: 0.34, bird: 0.29     uncertain
```

TENT assumes that unfamiliar/corrupted inputs often make a normally good classifier uncertain. It adjusts the model to make its predictions more decisive by minimizing **entropy**:

$$
H(p) = -\sum_c p_c \log p_c.
$$

Low entropy means one class gets most of the probability mass.

TENT wisely does not usually change the whole network. It updates the small scale/shift parameters in normalization layers. Think of it as adjusting the model's internal measuring instruments, not rewriting all of its knowledge.

### Why TENT became a foundation

It is simple, needs no source data or labels at deployment, and can work online. More importantly, it popularised two durable principles:

1. **Use a deployment-time objective when labels are absent.**
2. **Change a small parameter subset rather than the entire model.**

### Why blind confidence is dangerous

The uncertain classifier may be uncertain because the image truly is ambiguous—or because it is confidently about to make the wrong decision. Entropy minimization cannot distinguish these cases. Repeated updates can turn “not sure” into “very sure and very wrong.”

### Lesson for RealPDE

Velocity forecasting is regression, not classification. A velocity field is not a list of class probabilities, so there is no natural categorical entropy to minimize. Worse, trying to make continuous predictions artificially “sharp” may erase genuine turbulent variation, hurting TKE.

Borrow TENT's *discipline*—small, cheap updates—not its literal entropy loss.

---

## Act 4: “What if I only see one input?” — MEMO and valid augmentations

TENT often benefits from a batch of test inputs. But real systems may receive only one item at a time. [MEMO](https://arxiv.org/abs/2110.09506), by Zhang, Levine, and Finn, asks what to do then.

### The idea

Make several altered versions of the same input:

```text
one image → crop, colour shift, small distortion, ... → many views
```

The desired class should be the same across valid views. MEMO averages predictions over the views and makes that average confident.

$$
\bar p = \frac{1}{K}\sum_{i=1}^K p_\theta(y\mid a_i(x)),
\qquad L = H(\bar p).
$$

It therefore says: “give consistent answers to genuinely equivalent versions of this same observation.”

### The hidden assumption

An augmentation must preserve the real answer. A horizontal flip is often harmless for an object-category image. For air flowing around an airfoil, flipping or rotating a flow field can reverse the meaning of direction, boundary geometry, and wake location. It may produce a field that is visually plausible but physically a different problem.

### Lesson for RealPDE

Do not import computer-vision augmentations automatically. Any flow augmentation needs a physical argument. Tiny measurement-noise perturbations or carefully chosen temporal masking may be defensible; rotations and arbitrary crops generally are not.

Also remember the cost: $K$ views often mean $K$ forward passes. In RealPDE, every extra pass hurts the time score.

---

## Act 5: The model keeps learning—and begins forgetting

The early papers often studied one fixed shifted environment. Real deployment is harsher:

```text
morning conditions → afternoon conditions → camera noise → old conditions return
```

This is **continual test-time adaptation (CTTA)**. The model’s state persists along a stream.

### The failure: catastrophic forgetting

Suppose the model adapts strongly to a noisy patch of data. It shifts its weights toward explaining that patch. Later, when normal flow returns, it has partly forgotten the robust knowledge learned from thousands of source examples.

In RealPDE, the same thing can happen if we run SGD after every 20-frame window. Each window is small, noisy, and correlated with nearby windows. A single aggressive update seems harmless; hundreds of them compound.

### CoTTA: preserve a memory of the original student

[CoTTA](https://arxiv.org/abs/2203.13591) is a landmark continual-TTA method. It uses three connected ideas:

- an **EMA teacher**: a slowly changing copy of the model produces more stable pseudo-labels;
- prediction averaging over augmentations: do not trust one fragile view;
- **stochastic weight restoration**: occasionally restore a small random part of the adapted model to its original source-trained value.

The third idea is especially memorable. It is like allowing a student to learn from today’s lesson while periodically reopening parts of the original textbook so they do not drift too far.

### EATA: not every sample deserves an update

[EATA](https://arxiv.org/abs/2204.02610) makes another important observation: some samples are noisy, redundant, or unreliable. Updating on all of them is a choice, not a law.

It selects more reliable/non-redundant samples and uses a Fisher-information regularizer to discourage changes to parameters that seem important to the source model.

### RoTTA and memory

[RoTTA](https://arxiv.org/abs/2303.13899) focuses on realistic correlated streams. It uses robust normalization, a time-aware memory bank, and teacher–student learning. Its message is that neighbouring observations are not an independent, diverse batch merely because they arrive at different times.

### SAR: recover when adaptation is becoming unsafe

[SAR](https://arxiv.org/abs/2302.12400) studies “wild” online settings: mixed shifts, small batches, noise, and imbalance. It filters unreliable samples, prefers flatter/less fragile updates, and includes recovery logic for collapse.

### Lesson for RealPDE

These methods use pseudo-labels because they have no true labels. We do have delayed labels. So replace their uncertain pseudo-label quality checks with something better:

```text
previous target arrives
        ↓
measure actual previous prediction error
        ↓
decide whether and how much to update
```

The lasting CTTA lessons are:

- do not update blindly on every window;
- update a restricted state (adapter, calibration layer, normalization affine values) first;
- keep an anchor to the initial checkpoint;
- clip unusually large gradients;
- use recency carefully, but do not let one recent window erase everything;
- reset all adaptive state at a new trajectory boundary.

---

## Act 6: “Maybe we do not need to change the big model at all”

Some methods avoid backpropagation through the full model.

[T3A](https://papers.nips.cc/paper_files/paper/2021/hash/1415fe9fea0fa1e45dddcff5682239a0-Abstract.html) adjusts a small classifier/prototype state while freezing feature extraction. [LAME](https://arxiv.org/abs/2201.05718) changes predictions using relationships between samples rather than changing network weights.

The details are classification-specific, but the idea is powerful for RealPDE:

> A small correction state may be enough to fix a local deployment mismatch.

For example, write the prediction as:

$$
\hat y = f_{\theta_0}(x) + g_\phi(x),
$$

where $f_{\theta_0}$ is a frozen pretrained CNO/Transolver/FNO and $g_\phi$ is a small residual adapter. Only $\phi$ learns online.

This has three advantages:

1. the base physical forecaster stays intact;
2. the online optimizer is smaller and faster;
3. resetting a trajectory is simple: reset the adapter.

This is the most attractive first implementation direction for the competition.

---

## Act 7: The special challenge of physical forecasting

An image classifier makes one label. RealPDE predicts a 20-frame velocity movie over a grid. That difference changes what “good adaptation” means.

### Pointwise accuracy is not the entire story

You can make a prediction smoother and lower its pointwise noise, but then accidentally remove real turbulent fluctuations. The competition checks this using turbulent kinetic energy (TKE), which depends on variance through time.

You can have a decent average error over the whole field while still getting the wake behind the airfoil wrong. The competition checks selected wake probes using MVPE.

So a useful adaptation method must avoid the simplistic instinct to smooth everything.

### Physics can help, but it can also mislead

For an incompressible two-dimensional velocity field, we often expect approximately:

$$
\nabla \cdot \mathbf{u} = \frac{\partial u}{\partial x} + \frac{\partial v}{\partial y} \approx 0.
$$

This could be a diagnostic or a weak regularizer. But PIV is noisy, the airfoil creates a masked boundary, the grid has finite spacing, and a 2D measured slice may not reveal all 3D behaviour. A low divergence residual does not prove that the future wake is correct.

Use physics constraints as a seatbelt, not as the steering wheel, unless validation demonstrates a gain.

### A newer neural-operator direction

[Serrano et al. (2026)](https://arxiv.org/abs/2602.00884) explore test-time generalization for neural operators by searching over compositions of pretrained operators rather than changing weights. It is early and not the same delayed-label protocol, but it reinforces a useful point: **adaptation need not be gradient descent**. Test-time computation can mean choosing or composing fixed capabilities.

---

## Act 8: “How sure are you?” — uncertainty and safe predictions

An adaptation method can become overconfident. RealPDE explicitly rewards uncertainty intervals through the Safe Prediction Score (SPS).

Instead of only predicting:

```text
u = 0.42
```

the model can say:

```text
u = 0.42, and I expect the truth between 0.37 and 0.47.
```

There is a trade-off:

```text
very narrow interval → useful, but misses the truth often
very wide interval   → covers the truth, but says almost nothing useful
good interval        → covers at the promised rate and stays tight
```

**Conformal prediction** is a useful mental model. It calibrates a prediction interval from past errors instead of trusting a neural network’s confidence blindly. [Angelopoulos & Bates](https://arxiv.org/abs/2009.14193) is an accessible introduction.

For RealPDE, the natural first version is not a complicated ensemble. It is a causal residual tracker:

1. when the prior target arrives, measure $|y_{t-1}-\hat y_{t-1}|$;
2. retain a recent robust scale or quantile of those errors;
3. use it to set the current interval width;
4. reset at trajectory boundaries.

Classical conformal guarantees assume data are exchangeable (roughly: randomly ordered and similarly distributed). A drifting flow stream violates that assumption. The tracker is still useful, but call its coverage *empirical calibration*, not a magical formal guarantee.

---

## The research community’s cautionary ending

The story does not end with one universally best method. [TTAB: On Pitfalls of Test-Time Adaptation](https://arxiv.org/abs/2306.03536) tested methods across different shifts and found a sobering result: hyperparameters are hard to choose online, the starting model matters enormously, and no existing method handles every common shift.

That is healthy advice for this competition. Do not ask:

> “Which named TTA method should we copy?”

Ask instead:

> “What specific failure of our CNO/Transolver/FNO baseline do we observe on chronological real PIV trajectories, and what is the smallest causal mechanism that addresses it?”

---

## A practical RealPDE plan, now that you know the story

### Chapter 1: establish the protagonist

Start with the real-finetuned CNO checkpoint, frozen. Measure its five scores on held-out real trajectories. This tells us whether the main weakness is pointwise prediction, turbulence, wake probes, speed, or uncertainty.

### Chapter 2: make one small, reversible change

Add a small residual adapter or output calibration layer. Learn only from the previously revealed target. Keep the main pretrained model frozen.

### Chapter 3: teach the model when to stay quiet

Compare these policies on the exact same chronological validation trajectories:

```text
never update
update every window
update every k windows
update only when previous error crosses a threshold
update with a strict anchor/gradient clip
```

This is where continual-TTA ideas become concrete instead of decorative.

### Chapter 4: make uncertainty honest

Use past residuals to form intervals. Measure coverage and width as well as SPS. A narrow interval that frequently misses is not safe.

### Chapter 5: only then become ambitious

Try a physics regularizer, a teacher model, full-model updates, or a more complex controller only if the small method has a measured limitation. Every extra component costs debugging time and competition runtime.

---

## Reading path: ten papers, in the order that makes the field click

1. [Sun et al., Test-Time Training (2020)](https://arxiv.org/abs/1909.13231) — the original “learn from the input itself” idea.
2. [Wang et al., TENT (2021)](https://openreview.net/forum?id=uXl3bZLkr3c) — confidence-based adaptation and restricted parameters.
3. [Zhang, Levine & Finn, MEMO (2022)](https://arxiv.org/abs/2110.09506) — adaptation from one test input through valid augmentations.
4. [Wang et al., CoTTA (2022)](https://arxiv.org/abs/2203.13591) — continual adaptation, teacher models, and source-weight restoration.
5. [Niu et al., EATA (2022)](https://arxiv.org/abs/2204.02610) — select reliable updates and protect important parameters.
6. [Yuan, Xie & Li, RoTTA (2023)](https://arxiv.org/abs/2303.13899) — correlated real-world streams and memory.
7. [Niu et al., SAR (2023)](https://arxiv.org/abs/2302.12400) — stability under noisy, dynamic deployment.
8. [Zhao et al., TTAB pitfalls (2023)](https://arxiv.org/abs/2306.03536) — the necessary skeptical reading.
9. [Angelopoulos & Bates, Conformal Prediction (2021)](https://arxiv.org/abs/2009.14193) — calibrated uncertainty intervals.
10. [Xiao & Snoek survey (2024)](https://arxiv.org/abs/2411.03687) — the map when you want to explore beyond this guide.

---

## The one-paragraph takeaway

TTA began with the idea that a model can learn from the test input even without an answer. TENT made that idea lightweight; MEMO made it work from a single input; continual methods such as CoTTA, EATA, RoTTA, and SAR taught the field that naïve repeated updates are dangerous. RealPDE gives us delayed true targets, so we should use a direct supervised error rather than imitate classification entropy. The most promising first solution is therefore a small, anchored, causal adapter or calibration state that learns from the previous fully revealed flow window, updates only when evidence supports it, resets between trajectories, and tracks uncertainty from its own past errors.
