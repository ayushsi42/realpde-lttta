#!/usr/bin/env python3
"""Assemble and package a RealPDE Track 2 submission variant.

Scalable layout, so new ideas never touch this file:

    shared/                    <- never changes per-idea
        load_baseline.py
        rpde_baselines/
        checkpoints/<name>.pth
    variants/<idea_name>/      <- everything that DOES vary per idea
        submission.py
        policy.yaml            (optional, e.g. for agentic_rule)
        model.pth               (optional; or point at shared/checkpoints/...)
    build/<idea_name>/         <- generated staging dir (gitignored)
    build/<idea_name>.zip      <- generated, Codabench-ready

What this does, in order:
  1. Stages a variant: copies variants/<name>/* into build/<name>/, then
     copies in shared/load_baseline.py + shared/rpde_baselines/ ONLY IF the
     variant's submission.py actually imports load_baseline (so a submission
     with no real checkpoint stays tiny, like agentic_rule does today).
  2. If --checkpoint is given (a name under shared/checkpoints/, or a path),
     copies it into the staged dir as model.pth.
  3. Runs local_eval.py against the staged dir. Refuses to package on failure.
  4. Zips build/<name>/'s CONTENTS flat at the zip root (never the folder
     itself -- the #1 way Codabench zips fail).
  5. Checks the extracted size against the 256 MB cap.
  6. Re-extracts into a clean temp dir and re-imports it exactly the way the
     evaluator does (fresh module load, get_ttt_model / reset_ttt_state /
     ttt_step contract check), so the zip itself is verified, not just the
     staging directory.

Usage:
    python3 make_submission.py --variant agentic_rule
    python3 make_submission.py --variant baseline_reference
    python3 make_submission.py --variant my_new_idea --checkpoint sim_real_cno
    python3 make_submission.py --variant my_new_idea --checkpoint /path/to/model.pth

Adding a new idea:
    mkdir variants/my_new_idea
    cp variants/baseline_reference/submission.py variants/my_new_idea/
    # edit variants/my_new_idea/submission.py -- that's the only file to write
    python3 make_submission.py --variant my_new_idea
"""

from __future__ import annotations

import argparse
import importlib.util
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
SHARED = HERE / "shared"
VARIANTS = HERE / "variants"
BUILD = HERE / "build"
SIZE_CAP_BYTES = 256 * 1024 * 1024

EXCLUDE_DIR_NAMES = {"__pycache__", ".git", ".ipynb_checkpoints"}
EXCLUDE_SUFFIXES = {".pyc"}


def imports_load_baseline(submission_py: Path) -> bool:
    text = submission_py.read_text(encoding="utf-8", errors="ignore")
    return "load_baseline" in text


def copytree_filtered(src: Path, dst: Path) -> None:
    def ignore(dir_path: str, names: list[str]) -> set[str]:
        return {n for n in names if n in EXCLUDE_DIR_NAMES or Path(n).suffix in EXCLUDE_SUFFIXES}

    shutil.copytree(src, dst, ignore=ignore, dirs_exist_ok=True)


def stage_variant(name: str, checkpoint: str | None) -> Path:
    variant_dir = VARIANTS / name
    if not variant_dir.is_dir():
        raise SystemExit(f"[make_submission] no such variant: {variant_dir}")
    sub_file = variant_dir / "submission.py"
    if not sub_file.exists():
        raise SystemExit(f"[make_submission] {variant_dir} has no submission.py")

    staged = BUILD / name
    if staged.exists():
        shutil.rmtree(staged)
    staged.mkdir(parents=True)

    copytree_filtered(variant_dir, staged)
    print(f"[make_submission] staged variant files from {variant_dir}")

    if imports_load_baseline(sub_file):
        copytree_filtered(SHARED / "rpde_baselines", staged / "rpde_baselines")
        shutil.copy2(SHARED / "load_baseline.py", staged / "load_baseline.py")
        print("[make_submission] submission.py imports load_baseline -> "
              "copied shared/load_baseline.py + shared/rpde_baselines/")
    else:
        print("[make_submission] submission.py does not import load_baseline -> "
              "skipped shared model code (keeps the archive small)")

    if checkpoint:
        ckpt_path = Path(checkpoint)
        if not ckpt_path.exists():
            ckpt_path = SHARED / "checkpoints" / checkpoint
            if not ckpt_path.suffix:
                ckpt_path = ckpt_path.with_suffix(".pth")
        if not ckpt_path.exists():
            raise SystemExit(
                f"[make_submission] checkpoint not found: {checkpoint!r} "
                f"(looked at {checkpoint!r} and {SHARED / 'checkpoints'})"
            )
        shutil.copy2(ckpt_path, staged / "model.pth")
        print(f"[make_submission] copied checkpoint {ckpt_path} -> model.pth")
    elif (variant_dir / "model.pth").exists():
        print("[make_submission] variant already ships its own model.pth")

    return staged


def run_local_eval(staged_dir: Path) -> None:
    print(f"[make_submission] running local_eval.py --submission {staged_dir}")
    result = subprocess.run(
        [sys.executable, str(HERE / "local_eval.py"), "--submission", str(staged_dir)],
        cwd=str(HERE),
    )
    if result.returncode != 0:
        raise SystemExit(
            "[make_submission] local_eval.py failed -- fix the submission before "
            "packaging. Refusing to build a zip that would only fail on the platform."
        )
    print("[make_submission] local_eval.py OK")


def iter_package_files(staged_dir: Path):
    for path in sorted(staged_dir.rglob("*")):
        if path.is_dir():
            continue
        if any(part in EXCLUDE_DIR_NAMES for part in path.relative_to(staged_dir).parts):
            continue
        if path.suffix in EXCLUDE_SUFFIXES:
            continue
        yield path


def build_zip(staged_dir: Path, out_path: Path) -> list[str]:
    files = list(iter_package_files(staged_dir))
    if not files:
        raise SystemExit(f"[make_submission] no files found under {staged_dir}")
    if out_path.exists():
        out_path.unlink()
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in files:
            zf.write(path, path.relative_to(staged_dir))
    return [str(p.relative_to(staged_dir)) for p in files]


def check_extracted_size(zip_path: Path) -> None:
    total = sum(info.file_size for info in zipfile.ZipFile(zip_path).infolist())
    mb = total / (1024 * 1024)
    print(f"[make_submission] extracted size: {mb:.2f} MB (cap: 256 MB)")
    if total > SIZE_CAP_BYTES:
        raise SystemExit(
            f"[make_submission] extracted size {mb:.2f} MB exceeds the 256 MB cap. "
            "If this is a checkpoint, pack it with pack_ckpt_fp16.py first."
        )


def verify_zip_layout_and_contract(zip_path: Path) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(tmp_path)

        sub_file = tmp_path / "submission.py"
        if not sub_file.exists():
            raise SystemExit(
                "[make_submission] verification failed: submission.py is not at "
                "the extracted archive root."
            )

        sys.path.insert(0, str(tmp_path))
        try:
            spec = importlib.util.spec_from_file_location("packaged_submission", sub_file)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            if not hasattr(module, "get_ttt_model"):
                raise SystemExit(
                    "[make_submission] verification failed: no get_ttt_model()."
                )
            model = module.get_ttt_model(str(tmp_path), "cpu")
            for attr in ("reset_ttt_state", "ttt_step"):
                if not hasattr(model, attr):
                    raise SystemExit(
                        f"[make_submission] verification failed: missing {attr}()."
                    )

            import torch

            model.reset_ttt_state()
            x = torch.randn(1, 20, 32, 64, 3)
            pred, info = model.ttt_step(x, None)
            if tuple(pred.shape) != (1, 20, 32, 64, 3):
                raise SystemExit(
                    f"[make_submission] verification failed: ttt_step shape "
                    f"{tuple(pred.shape)} != (1, 20, 32, 64, 3)."
                )
            if not isinstance(info, dict) or "adapt_loss" not in info:
                raise SystemExit(
                    "[make_submission] verification failed: info missing 'adapt_loss'."
                )
            pred2, _info2 = model.ttt_step(x, x)
            if tuple(pred2.shape) != (1, 20, 32, 64, 3):
                raise SystemExit(
                    "[make_submission] verification failed: second-call shape wrong."
                )
            print("[make_submission] zip-root import + interface contract OK")
        finally:
            sys.path.remove(str(tmp_path))
            sys.modules.pop("packaged_submission", None)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", required=True, help="name of a folder under variants/")
    ap.add_argument("--checkpoint", default=None,
                    help="checkpoint name under shared/checkpoints/ (or a full path); "
                         "copied into the staged submission as model.pth")
    ap.add_argument("--skip-local-eval", action="store_true")
    args = ap.parse_args()

    staged = stage_variant(args.variant, args.checkpoint)

    if not args.skip_local_eval:
        run_local_eval(staged)
    else:
        print("[make_submission] --skip-local-eval set, skipping smoke test")

    out_path = BUILD / f"{args.variant}.zip"
    packaged = build_zip(staged, out_path)
    print(f"[make_submission] packaged {len(packaged)} files into {out_path}")
    for name in packaged:
        print(f"[make_submission]   + {name}")

    check_extracted_size(out_path)
    verify_zip_layout_and_contract(out_path)

    print(f"[make_submission] DONE: {out_path} is ready to upload to Codabench.")


if __name__ == "__main__":
    main()
