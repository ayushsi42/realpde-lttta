# RealPDE Track 2 LTTTA Starting Kit v6

This kit contains the minimal files to build a Codabench submission for Track 2:
Long-Term Test-Time Adaptation on streaming real-world PIV data.

> **Project layout:** this follows a conventional ML layout. Edit only
> `src/solution/`; use Git branches for experiments. `data/` and `checkpoints/`
> are local and ignored by Git, while `outputs/` holds generated artifacts.

```text
src/          Python source: active solution, baseline loader, vendored models
data/         local example and training PIV data (Git-ignored)
checkpoints/  local pretrained weights (Git-ignored)
scripts/      run evaluation, package a ZIP, visualize data, pack checkpoints
docs/         task contract, metrics, study notes, literature review
outputs/      generated submission ZIPs and visualizations (Git-ignored)
```

## Files

- `src/solution/`: the active submission: edit `submission.py` and its
  optional `policy.yaml` as the project advances.
- `src/ttt_model.py`: optional convenience base class for the TTT interface.
- `scripts/local_eval.py`: run a CPU smoke test against the bundled `data/example`, and
  report the real subscores using the bundled `scoring.py`.
- `scripts/scoring.py`: the official scoring program (the exact leaderboard formulas).
  Run it yourself on real data, or let `scripts/local_eval.py` call it.
- `scripts/pack_ckpt_fp16.py`: pack an fp32 checkpoint to fp16 to fit the size limit.
- `src/load_baseline.py`: build CNO / FNO / Transolver and load an official
  checkpoint, ready to wrap as your TTT base model (see "Baseline Models").
- `src/rpde_baselines/`: vendored baseline model code (renamed so it never shadows
  the evaluator's own `realpdebench` package); imports offline.
- `src/solution/`: currently starts from a bounded-controller adaptation of
  [agentic_LTTTA](https://github.com/PgUpDn/agentic_LTTTA); see
  `src/solution/README.md`.
- `docs/interface.md`, `docs/metrics.md`: interface contract and metric summary.
- `docs/literature_review_test_time_adaptation.md`: curated background reading
  on test-time adaptation and its relevance to this track.
- `scripts/visualize_piv_window.py`: generates an interactive PIV window explorer
  under `artifacts/` (generated files are not versioned).
- `data/example/`: two tiny synthetic trajectories + the official normalization
  stats, for local shape checks only (regenerate with `data/example/make_example.py`).
- `data/`: local downloaded training data only; intentionally ignored by Git.
  `train_real.tar.gz` is a 7.39 GB download, so keep it local and extract/use it
  from this directory rather than placing it in a submission archive.

## Interface

`submission.py` must define:

```python
def get_ttt_model(submission_dir, device):
    ...  # return an object with reset_ttt_state() and ttt_step()
```

The returned object implements:

- `reset_ttt_state()`: called at each trajectory start; restore checkpoint
  weights, clear adaptation state.
- `ttt_step(input_norm, prev_target_norm=None)`: called once per step, batch
  size 1, on `(1, 20, 32, 64, 3)` normalized tensors. `prev_target_norm` is the
  ground truth of the previous step (`None` on the first step). Return
  `(pred_norm, info)`; the evaluator reads `info["adapt_loss"]` (a python float
  or `None`) and, optionally, `info["lower"]` / `info["upper"]` for the Safe
  Prediction Score. Any other key is ignored.

Evaluation is duck-typed, so subclassing `ttt_model.py` is optional. See
`docs/interface.md` for the full contract.

## Local Smoke Test

`scripts/local_eval.py` mirrors the official streaming loop (trajectory resets,
prev-target passing, per-step timing) on the bundled example data, then feeds the
predictions through the bundled `scoring.py` to print the five real subscores. It
uses the same subscore formulas the leaderboard uses, but the numbers on the tiny
synthetic example data are illustrative only: point `--data` at real downloaded
data for comparable scores, and note `time_score` reflects your local wall time.

The leaderboard combines them into a single `final_score`; that combination is
not published, so no total is printed here.

```bash
# run the active solution directly against the local evaluator
python scripts/local_eval.py

# or build + verify a Codabench-ready zip in one step (recommended)
python scripts/make_submission.py
```

`scripts/local_eval.py` mirrors the evaluator's streaming loop (the evaluator itself is
not downloadable); for the exact submission contract, `docs/interface.md` is
authoritative.

## Evaluation Sandbox

Your model runs out-of-process in an isolated subprocess. It reads your
submission archive and the Python runtime normally, writes freely (temp files,
caches, checkpoints), and uses torch / CUDA / numpy as usual. Only the hidden
evaluation data (ground truth) is unreadable from disk: any attempt to open it
is denied by the kernel. You never need it, because `ttt_step` is handed the
normalized input and previous target directly. Of the `info` dict you return,
the evaluator keeps `adapt_loss` and the optional `lower` / `upper` bounds; put
nothing else load-bearing in it.

## Timing

Everything inside `ttt_step` is timed: adaptation on the previous pair,
controller logic, optional LLM round-trips, and the forward pass. `reset_ttt_state`
is timed too, charged to the step after the boundary, so moving work between the
two changes nothing. Model construction and checkpoint loading are **not** timed;
they run once before the stream. The Time score uses the mean per-step wall time. A separate 10-minute wall-clock
execution limit covers the whole run (loading included), so keep loading modest.

## Submission Size

The extracted archive (checkpoint included) must stay under **256 MB**. If your
fp32 checkpoint is too large, pack it to fp16:

```bash
python scripts/pack_ckpt_fp16.py model_fp32.pth model.pth
```

The tool handles complex tensors safely; see its docstring for the matching
`unpack_fp16` load helper.

## Baseline Models (pretrained checkpoints)

The kit vendors the official baseline model code (`rpde_baselines/`: CNO, FNO,
Transolver) plus `load_baseline.py`. `einops` (needed by the Transolver
forward) is already installed in the Track 2 evaluation image, so there is
nothing extra to ship.

Checkpoints are on the competition Google Drive (folder `baseline_checkpoints/`
inside the data release, id `1Cg23DoTuSvWXR3Mm1uRfmMNAbkyaIhrQ`; not shipped in
this kit):

```
baseline_checkpoints/
├── sim_pretrain/          # sim-only pretraining
│   ├── sim_cno.pth
│   ├── sim_fno.pth
│   ├── sim_fno_fp16.pth   # fp16-packed FNO (fits the 256 MB cap)
│   └── sim_transolver.pth
└── sim_real_ft/           # fine-tuned on real PIV data (best starting point)
    ├── sim_real_cno.pth
    ├── sim_real_fno.pth
    ├── sim_real_fno_fp16.pth
    └── sim_real_transolver.pth
```

Sizes: CNO 32 MB, Transolver 50 MB, FNO 403 MB (fp32) / 201 MB (fp16-packed).
Only FNO needs fp16 packing to fit under the 256 MB submission cap.

Use one as the base model your TTT method adapts:

```python
from load_baseline import load_baseline, make_example_input
from solution.submission import AgenticTTTModel

base, meta = load_baseline("sim_real_cno.pth")     # type auto-detected
model = AgenticTTTModel(base, device="cpu", policy={...})  # adapt in ttt_step

x = make_example_input("data/example/test_real/5025_5.h5")  # (1,20,32,64,3)
y = base(x)                                                 # (1,20,32,64,3)
```

The example call is a raw-space shape check; during evaluation `ttt_step`
receives tensors already normalized with the official stats, which is the space
the checkpoints were trained in. Geometry: channels `[u, v, p]` (p = 0),
`T_in = T_out = 20`, eval resolution `32 x 64` (the kit example h5 stores raw
`64 x 128`; `make_example_input(..., sub_s=2)` handles the subsampling).

## Training Data

The example data here is for shape checks only. The full training release
(simulated pretraining + real finetuning trajectories) is on Google Drive:
[NeurIPS 2026 RealPDE Competition data](https://drive.google.com/drive/folders/1Cg23DoTuSvWXR3Mm1uRfmMNAbkyaIhrQ).
Track 1 and Track 2 share the same release.

## LLM Access

Optional LLM calls must go through the organizer-provided gateway. The endpoint,
the model name and a code example are on the competition Submission page. Direct
network access and your own API keys are prohibited, and LLM latency counts
toward Time. The numbers in `src/solution/policy.yaml` are the active solution's own, not
gateway limits.

## Active Solution Starting Point

`src/solution/` is a submittable adaptation of the agentic-LTTTA baseline
([github.com/PgUpDn/agentic_LTTTA](https://github.com/PgUpDn/agentic_LTTTA)):
a bounded controller (skip / recalibrate / update-adapter) driven by an
offline-tuned `policy.yaml`, with optional online LLM action selection through
the organizer gateway (auto-detected from the injected `OPENAI_*` environment,
safe fallback to the rule policy). Run `python scripts/local_eval.py --submission
src/solution`, and see `src/solution/README.md` for how it maps the upstream
offline-design / online-execution demos onto the official Track 2 protocol.
