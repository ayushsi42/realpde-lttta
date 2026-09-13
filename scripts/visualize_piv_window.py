#!/usr/bin/env python3
"""Create an interactive browser view of one real-format PIV trajectory.

Example:
    python3 scripts/visualize_piv_window.py \
      --input data/example/test_real/5025_5.h5 \
      --output outputs/visualizations/piv_window_explorer.html

The generated page is self-contained: it embeds Plotly and the downsampled
velocity fields, so it can be opened directly in a browser without a server.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch


IN_STEP = 20
OUT_STEP = 20
STRIDE = 20
SUBSAMPLE = 2


def window_ranges(n_frames: int) -> list[tuple[int, int, int, int]]:
    """Return (input_start, input_end, target_start, target_end) windows."""
    return [
        (start, start + IN_STEP - 1, start + IN_STEP, start + IN_STEP + OUT_STEP - 1)
        for start in range(0, n_frames - (IN_STEP + OUT_STEP) + 1, STRIDE)
    ]


def heatmap(z: np.ndarray, title: str, colorscale: str, zmin: float, zmax: float):
    return go.Heatmap(
        z=z,
        colorscale=colorscale,
        zmin=zmin,
        zmax=zmax,
        colorbar={"title": title, "len": 0.78},
        hovertemplate="grid x=%{x}, y=%{y}<br>value=%{z:.4f}<extra></extra>",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Input PIV .h5 trajectory")
    parser.add_argument("--output", type=Path, required=True, help="Output .html file")
    args = parser.parse_args()

    with h5py.File(args.input, "r") as data:
        u = np.asarray(data["u"][:, ::SUBSAMPLE, ::SUBSAMPLE], dtype=np.float32)
        v = np.asarray(data["v"][:, ::SUBSAMPLE, ::SUBSAMPLE], dtype=np.float32)

    # These are the exact statistics used by the local evaluation pipeline.
    # Input and target have separately stored statistics, though their values
    # are nearly identical for this dataset.
    stats_path = args.input.parent.parent / "mean_std_real.pt"
    mean_in, mean_tgt, std_in, std_tgt = torch.load(stats_path, weights_only=False)
    mean_in, mean_tgt = mean_in.numpy(), mean_tgt.numpy()
    std_in, std_tgt = std_in.numpy(), std_tgt.numpy()
    safe_std_in = np.where(std_in == 0, 1.0, std_in)
    safe_std_tgt = np.where(std_tgt == 0, 1.0, std_tgt)

    n_frames, height, width = u.shape
    speed = np.hypot(u, v)
    ranges = window_ranges(n_frames)
    u_abs = float(np.max(np.abs(u)))
    v_abs = float(np.max(np.abs(v)))
    # A real scalar read from the first retained PIV grid location.  It makes
    # the normalization equation tangible without pretending it is a model
    # prediction.
    raw_u, raw_v = float(u[0, 0, 0]), float(v[0, 0, 0])
    norm_u = (raw_u - float(mean_in[0])) / float(safe_std_in[0])
    norm_v = (raw_v - float(mean_in[1])) / float(safe_std_in[1])

    figure = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=("Horizontal velocity <i>u</i>", "Vertical velocity <i>v</i>", "Speed √(u² + v²)"),
        horizontal_spacing=0.075,
    )
    figure.add_trace(heatmap(u[0], "u", "RdBu_r", -u_abs, u_abs), row=1, col=1)
    figure.add_trace(heatmap(v[0], "v", "RdBu_r", -v_abs, v_abs), row=1, col=2)
    figure.add_trace(heatmap(speed[0], "speed", "Viridis", 0, float(speed.max())), row=1, col=3)

    steps = []
    for frame in range(n_frames):
        phase = "INPUT" if frame % 40 < 20 else "TARGET"
        sample = frame // 20
        steps.append(
            {
                "method": "update",
                "label": str(frame),
                "args": [
                    {"z": [u[frame], v[frame], speed[frame]]},
                    {
                        "title": {
                            "text": (
                                f"<b>{args.input.name}</b> &nbsp;•&nbsp; physical frame {frame} of {n_frames - 1} "
                                f"&nbsp;•&nbsp; <span style='color:#59d4ff'>{phase}</span> region"
                            )
                        }
                    },
                ],
            }
        )

    timeline_shapes = []
    annotations = []
    for index, (i0, i1, t0, t1) in enumerate(ranges):
        y0 = 0.62 if index % 2 == 0 else 0.10
        timeline_shapes.extend(
            [
                {"type": "rect", "xref": "x", "yref": "y", "x0": i0, "x1": i1, "y0": y0, "y1": y0 + .22,
                 "fillcolor": "#2276d2", "line": {"width": 0}, "opacity": .95},
                {"type": "rect", "xref": "x", "yref": "y", "x0": t0, "x1": t1, "y0": y0, "y1": y0 + .22,
                 "fillcolor": "#e9933f", "line": {"width": 0}, "opacity": .95},
            ]
        )
        annotations.append({"x": (i0 + i1) / 2, "y": y0 + .11, "xref": "x", "yref": "y", "text": f"sample {index}<br>input", "showarrow": False, "font": {"color": "white", "size": 11}})
        annotations.append({"x": (t0 + t1) / 2, "y": y0 + .11, "xref": "x", "yref": "y", "text": f"sample {index}<br>target", "showarrow": False, "font": {"color": "white", "size": 11}})

    figure.update_layout(
        title={"text": f"<b>{args.input.name}</b> &nbsp;•&nbsp; physical frame 0 of {n_frames - 1} &nbsp;•&nbsp; <span style='color:#59d4ff'>INPUT</span> region", "x": .5},
        template="plotly_dark",
        paper_bgcolor="#07111f",
        plot_bgcolor="#07111f",
        height=610,
        margin={"l": 30, "r": 30, "t": 95, "b": 115},
        sliders=[{"active": 0, "currentvalue": {"prefix": "Show physical frame: ", "font": {"size": 16}}, "pad": {"t": 46}, "steps": steps}],
    )
    figure.update_xaxes(title_text="grid x", showgrid=False)
    figure.update_yaxes(title_text="grid y", showgrid=False, autorange="reversed")

    figure.add_annotation(
        xref="paper", yref="paper", x=.5, y=-.40, showarrow=False, align="center",
        text=(
            "<b>How this stream is split</b><br>"
            "Blue = model input (known past); orange = target (future to predict). "
            "The next sample begins where the previous target ended."
        ),
        font={"size": 14, "color": "#cbd5e1"},
    )

    timeline = go.Figure()
    timeline.update_layout(
        template="plotly_dark", paper_bgcolor="#07111f", plot_bgcolor="#07111f", height=235,
        margin={"l": 58, "r": 22, "t": 20, "b": 42}, shapes=timeline_shapes, annotations=annotations,
        xaxis={"title": "Physical time-frame index", "range": [-1, n_frames], "dtick": 5, "showgrid": True, "gridcolor": "#1e3047"},
        yaxis={"visible": False, "range": [0, 1]},
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    page = f"""<!doctype html><html><head><meta charset='utf-8'><title>PIV Window Explorer</title>
    <style>body{{margin:0;background:#07111f;color:#e5edf8;font:16px system-ui,-apple-system,sans-serif}}main{{max-width:1450px;margin:auto;padding:28px}}h1,h2{{margin:0 0 8px}}h2{{margin-top:48px;font-size:25px}}p,li{{color:#b7c5d8;line-height:1.55}}.cards{{display:flex;gap:14px;flex-wrap:wrap;margin:20px 0}}.card,.box{{background:#0c1a2d;border:1px solid #1c3553;border-radius:12px;padding:14px 18px;min-width:180px}}.card b{{display:block;color:#69d6ff;font-size:20px;margin-bottom:4px}}.legend span{{display:inline-block;padding:4px 10px;border-radius:6px;margin-right:7px;color:white}}.input{{background:#2276d2}}.target{{background:#e9933f}}.pipeline{{display:grid;grid-template-columns:repeat(5,minmax(170px,1fr));gap:12px;align-items:stretch;margin:20px 0}}.pipe{{position:relative;background:#0c1a2d;border:1px solid #28517e;border-radius:14px;padding:17px}}.pipe:not(:last-child):after{{content:'→';position:absolute;right:-11px;top:40%;font-size:25px;color:#69d6ff;z-index:2}}.pipe h3{{margin:0 0 8px;color:#72dcff;font-size:16px}}.pipe code,.math{{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;color:#f8c46c;font-size:13px;word-break:break-word}}.split{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}.note{{border-left:3px solid #e9933f;padding:8px 13px;background:#101d2d}}@media(max-width:900px){{.pipeline{{grid-template-columns:1fr}}.pipe:after{{display:none}}.split{{grid-template-columns:1fr}}}}</style></head><body><main>
    <h1>PIV flow-window explorer</h1><p>This is the exact bundled trajectory <code>{args.input.name}</code>, spatially downsampled from {height * 2} × {width * 2} to {height} × {width}, exactly as evaluation does. Move the slider to see a real measured velocity field at each physical time.</p>
    <div class='cards'><div class='card'><b>{n_frames}</b>physical frames in this file</div><div class='card'><b>{len(ranges)}</b>valid 20-in → 20-out samples</div><div class='card'><b>20 + 20</b>frames per prediction task</div><div class='card'><b>[u, v, p]</b>channels; p is zero here</div></div>
    {figure.to_html(full_html=False, include_plotlyjs=True)}
    <div class='legend'><span class='input'>Known input frames</span><span class='target'>Future target frames</span></div>
    <p><b>Important:</b> one data point is not “only 40 frames forever.” A <em>training/evaluation sample</em> is a 40-frame slice: 20 known frames plus the next 20 future frames. A trajectory continues, and the evaluator advances by 20 frames to make the next sample.</p>
    {timeline.to_html(full_html=False, include_plotlyjs=False)}
    <h2>Every data operation, in order</h2>
    <p>This is the exact evaluation data path. The model never receives the native 64 × 128 field; it receives the downsampled and normalized version.</p>
    <div class='pipeline'>
      <div class='pipe'><h3>1. Raw PIV trajectory</h3><code>u, v, p<br>[60, 64, 128, 3]</code><p>Physical velocity measurements. Here <code>p = 0</code>.</p></div>
      <div class='pipe'><h3>2. Choose one window</h3><code>step 0<br>input = 0:20<br>target = 20:40</code><p>Two 20-frame blocks are cut from the continuing trajectory.</p></div>
      <div class='pipe'><h3>3. Downsample space</h3><code>x[:, ::2, ::2, :]<br>[20, 64, 128, 3]<br>→ [20, 32, 64, 3]</code><p>Keep grid rows 0, 2, 4… and columns 0, 2, 4….</p></div>
      <div class='pipe'><h3>4. Normalize</h3><code>x_norm = (x − μ) / σ</code><p>Numbers are rescaled into a model-friendly range, separately for each channel.</p></div>
      <div class='pipe'><h3>5. Neural forecast</h3><code>ŷ_norm = Fθ(x_norm)<br>[1,20,32,64,3]</code><p>Predict 20 future frames. θ are the learned checkpoint weights.</p></div>
    </div>
    <div class='pipeline'>
      <div class='pipe'><h3>6. Denormalize output</h3><code>ŷ = ŷ_norm × σ_target + μ_target</code><p>The evaluator converts the prediction back to physical units before scoring.</p></div>
      <div class='pipe'><h3>7. Score against truth</h3><code>ŷ vs target<br>on u and v</code><p>Accuracy, turbulence, wake probes, speed, and uncertainty are scored.</p></div>
      <div class='pipe'><h3>8. One step later</h3><code>previous target<br>is revealed</code><p>The prior target becomes legal feedback only after its prediction was returned.</p></div>
      <div class='pipe'><h3>9. Optional adaptation</h3><code>loss = mean((Fθ(x_prev) − y_prev)²)<br>θ ← θ − lr × ∇θ loss</code><p>Use delayed feedback to adjust weights carefully.</p></div>
      <div class='pipe'><h3>10. Repeat / reset</h3><code>next window<br>or new trajectory</code><p>At a new trajectory boundary, reset back to checkpoint weights.</p></div>
    </div>
    <div class='split'>
      <div class='box'><h3>Why downsample?</h3><p>Native PIV is <code>64 × 128 = 8,192</code> locations per frame. Evaluation retains every other point in each direction: <code>32 × 64 = 2,048</code> locations—one quarter as many. This reduces memory and computation by about 4× and makes every submission use the same spatial resolution. It is not averaging: the code uses array slicing <code>::2</code>, so it keeps every second measurement.</p><p>Using the native array directly would return the wrong tensor shape and would not match the pretrained checkpoints or evaluator.</p></div>
      <div class='box'><h3>Exact normalization constants in this kit</h3><p>For input channels <code>[u, v, p]</code>:</p><div class='math'>μ_in = [{mean_in[0]:.8f}, {mean_in[1]:.8f}, {mean_in[2]:.1f}]<br>σ_in = [{std_in[0]:.8f}, {std_in[1]:.8f}, 1.0*]</div><p>For targets:</p><div class='math'>μ_target = [{mean_tgt[0]:.8f}, {mean_tgt[1]:.8f}, {mean_tgt[2]:.1f}]<br>σ_target = [{std_tgt[0]:.8f}, {std_tgt[1]:.8f}, 1.0*]</div><p>*The stored pressure standard deviation is zero; the loader safely substitutes 1 to avoid dividing by zero. Pressure itself remains zero.</p></div>
    </div>
    <div class='split'>
      <div class='box'><h3>A real scalar, calculated exactly</h3><p>At physical frame 0 and retained grid point <code>(y=0, x=0)</code>, this file contains:</p><div class='math'>u_raw = {raw_u:.8f}<br>u_norm = ({raw_u:.8f} − {mean_in[0]:.8f}) / {safe_std_in[0]:.8f}<br>       = {norm_u:.8f}<br><br>v_raw = {raw_v:.8f}<br>v_norm = ({raw_v:.8f} − ({mean_in[1]:.8f})) / {safe_std_in[1]:.8f}<br>       = {norm_v:.8f}</div><p>This operation is applied independently at every time, row, column, and channel—<code>20 × 32 × 64 × 3 = 122,880</code> values per input window (then a batch dimension of 1 is added).</p></div>
      <div class='box'><h3>What “matrix multiplication” means here</h3><p>There is no single universal matrix multiplication to display because CNO, FNO, and Transolver use different internals, and their learned weights come from the checkpoint you choose. The exact common operation is:</p><div class='math'>ŷ_norm = Fθ(x_norm)</div><p>For the included <code>TinyForecaster</code> demonstration model, <code>Fθ</code> is two 3×3 convolutions with ReLU and a residual addition:</p><div class='math'>z₁ = Conv3×3(x; W₁,b₁)<br>z₂ = ReLU(z₁)<br>ŷ = x + Conv3×3(z₂; W₂,b₂)</div><p>At one output pixel, one convolution value is a weighted sum: <code>Σ(channel,dy,dx) W[channel,dy,dx] × x[channel,y+dy,x+dx] + b</code>. Real checkpoint weights determine every number in that sum; the placeholder weights are random, so they are not meaningful physics.</p></div>
    </div>
    <p class='note'><b>Do not mix the two operations:</b> downsampling changes the number of spatial locations; normalization changes the numerical scale of each retained value. They are separate steps. The model operates after both.</p>
    </main></body></html>"""
    args.output.write_text(page, encoding="utf-8")
    print(f"Wrote {args.output} ({len(ranges)} valid windows from {n_frames} frames).")


if __name__ == "__main__":
    main()
