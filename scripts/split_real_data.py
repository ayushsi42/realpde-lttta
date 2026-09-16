#!/usr/bin/env python3
"""Chronological, trajectory-level train/val splitter for the real PIV archive.

Each file under ``data/train_real/train_real/*.h5`` is one *independent*
wind-tunnel/PIV trajectory named ``{re}_{aoa}.h5`` (Reynolds number, angle of
attack in degrees; confirmed by inspecting the ``re``/``aoa`` scalar datasets
inside the files). There is no shared wall-clock timeline across files -- each
trajectory's own ``t`` array restarts at 0 -- so there is no literal
"timestamp" to sort by. To still produce a deterministic, non-random,
order-preserving split (never a `random.shuffle`/`train_test_split` over
files) that guards against leakage, this script sorts trajectories by
``(re, aoa)`` ascending -- the natural, reproducible axis along which this
factorial sweep was designed -- and holds out a contiguous block from the
*end* of that order (the highest-Re regimes) as validation. This mirrors a
forward/extrapolation-style split: validation is drawn from the "far" end of
the design sweep, never interleaved at random with training trajectories.

Whichever axis you consider "time", the property that matters for this
competition's leakage concern is enforced regardless: splitting happens at
the whole-trajectory level, so no window from a held-out trajectory's file
can appear in anything used for training/threshold-picking, and no window
from a training trajectory can leak into validation. This is different from
(and safer than) a random split over individual (file, time_id) windows,
which would let near-duplicate adjacent windows from the same trajectory land
on both sides of the split.

Usage
-----
    python3 scripts/split_real_data.py \\
        --data-dir data/train_real/train_real \\
        --held-out-re 24150 25425 26700 \\
        --val-link data/train_real_holdout

Writes:
    outputs/real_data_split.json   -- manifest: train/val file lists + rationale
    <val-link>/test_real/*.h5      -- symlinks to the held-out trajectory files
    <val-link>/mean_std_real.pt    -- copy of the official real-data norm stats

The symlink directory matches the layout ``scripts/local_eval.py`` expects
(``<data>/test_real/*.h5`` + ``<data>/mean_std_real.pt``), so the held-out
split can be evaluated directly with:

    python3 scripts/local_eval.py --data data/train_real_holdout ...
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_STATS = ROOT / "data" / "example" / "mean_std_real.pt"


def parse_trajectories(data_dir: Path) -> list[tuple[int, int, str]]:
    """Return sorted (re, aoa, filename-stem) for every ``*.h5`` under data_dir."""
    out = []
    for p in sorted(data_dir.glob("*.h5")):
        stem = p.stem  # "{re}_{aoa}"
        try:
            re_str, aoa_str = stem.split("_")
            out.append((int(re_str), int(aoa_str), stem))
        except ValueError:
            raise SystemExit(f"Unexpected filename (want '{{re}}_{{aoa}}.h5'): {p.name}")
    out.sort()
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=str(ROOT / "data" / "train_real" / "train_real"),
                     help="directory of extracted train_real/*.h5 files")
    ap.add_argument("--held-out-re", type=int, nargs="*", default=[24150, 25425, 26700],
                     help="Reynolds-number groups (the highest-Re tail of the sweep) "
                          "to hold out entirely for validation")
    ap.add_argument("--held-out-count", type=int, default=None,
                     help="alternative to --held-out-re: hold out the last N "
                          "trajectories in sorted (re, aoa) order instead of naming "
                          "explicit re groups")
    ap.add_argument("--val-link", default=str(ROOT / "data" / "train_real_holdout"),
                     help="output directory to populate with symlinks (test_real/) "
                          "matching the local_eval data-dir layout")
    ap.add_argument("--manifest", default=str(ROOT / "outputs" / "real_data_split.json"),
                     help="where to write the split manifest JSON")
    ap.add_argument("--stats-src", default=str(DEFAULT_STATS),
                     help="official real-data mean/std stats to copy alongside the split")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        raise SystemExit(f"Not a directory: {data_dir}")

    trajectories = parse_trajectories(data_dir)
    if not trajectories:
        raise SystemExit(f"No .h5 files found under {data_dir}")

    if args.held_out_count is not None:
        held_out = set(t[2] for t in trajectories[-args.held_out_count:])
        rationale = (f"last {args.held_out_count} trajectories in ascending "
                      "(re, aoa) sort order")
    else:
        held_out_re = set(args.held_out_re)
        held_out = set(t[2] for t in trajectories if t[0] in held_out_re)
        rationale = f"all trajectories with re in {sorted(held_out_re)} (highest-Re tail)"

    train = [t for t in trajectories if t[2] not in held_out]
    val = [t for t in trajectories if t[2] in held_out]
    if not val:
        raise SystemExit("Held-out selection matched zero trajectories; check --held-out-re.")

    manifest = {
        "source_dir": str(data_dir),
        "n_total_trajectories": len(trajectories),
        "n_train_trajectories": len(train),
        "n_val_trajectories": len(val),
        "split_rule": (
            "Deterministic, non-random, whole-trajectory split. Trajectories sorted "
            "ascending by (re, aoa) -- the factorial-sweep design axis, since each "
            "file is an independent run with its own restarting time axis and no "
            "shared wall-clock timestamp is stored. Held-out validation is a "
            "contiguous block from the high end of that order (" + rationale + "), "
            "never a random subset, so no window from any held-out file leaks into "
            "training/threshold-picking and vice versa."
        ),
        "held_out_rationale": rationale,
        "train_trajectories": [f"{re_}_{aoa}" for re_, aoa, _ in train],
        "val_trajectories": [f"{re_}_{aoa}" for re_, aoa, _ in val],
    }

    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"[split_real_data] wrote manifest: {manifest_path}")
    print(f"[split_real_data] {len(train)} train / {len(val)} val trajectories "
          f"(of {len(trajectories)} total)")
    print(f"[split_real_data] val: {manifest['val_trajectories']}")

    val_link = Path(args.val_link)
    val_test_dir = val_link / "test_real"
    val_test_dir.mkdir(parents=True, exist_ok=True)
    # Clear stale symlinks from a previous run before relinking.
    for existing in val_test_dir.glob("*.h5"):
        if existing.is_symlink() or existing.is_file():
            existing.unlink()
    for re_, aoa, stem in val:
        src = (data_dir / f"{stem}.h5").resolve()
        dst = val_test_dir / f"{stem}.h5"
        os.symlink(src, dst)

    stats_src = Path(args.stats_src)
    if stats_src.exists():
        shutil.copy(stats_src, val_link / "mean_std_real.pt")
        print(f"[split_real_data] copied norm stats from {stats_src}")
    else:
        print(f"[split_real_data] WARNING: stats source {stats_src} not found; "
              f"copy mean_std_real.pt into {val_link} manually before evaluating.")

    print(f"[split_real_data] held-out eval dir ready: {val_link} "
          f"(test_real/*.h5 symlinks + mean_std_real.pt)")


if __name__ == "__main__":
    main()
