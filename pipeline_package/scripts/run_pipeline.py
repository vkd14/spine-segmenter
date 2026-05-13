#!/usr/bin/env python3
"""
Integrated nnU-Net pipeline with:
  - Per-fold training launched as subprocess
  - External early stopping (monitors EMA Dice; kills when plateau detected)
  - Post-processing after each fold (largest CC + sagittal strip + order check)
  - Warm-start each subsequent fold from the previous fold's best checkpoint
  - Automatic fold sequencing (0 → 1 → 2 → 3 → 4)

Usage:
    # Start from fold 1 (fold 0 already running/done), max 200 ep, patience 80
    python scripts/run_pipeline.py --start-fold 1 --max-epochs 200 --patience 80

    # Full run all folds from scratch
    python scripts/run_pipeline.py --start-fold 0 --max-epochs 1000 --patience 150

    # Resume (skips folds that already have checkpoint_best.pth)
    python scripts/run_pipeline.py --start-fold 0 --resume
"""

import os
import re
import sys
import time
import signal
import argparse
import subprocess
import glob
from pathlib import Path
from datetime import datetime, timedelta


def kill_process_group(proc):
    """Kill the entire process group (main proc + all workers)."""
    try:
        pgid = os.getpgid(proc.pid)
        os.killpg(pgid, signal.SIGTERM)
        time.sleep(5)
        try:
            os.killpg(pgid, signal.SIGKILL)  # force-kill any survivors
        except ProcessLookupError:
            pass
    except ProcessLookupError:
        pass  # already gone
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent.resolve()
VENV_BIN = ROOT / ".venv" / "bin"

DATASET_ID = 100
DATASET_NAME = "Dataset100_SpineL1L5"
TRAINER = "nnUNetTrainer"
PLANS_ID = "nnUNetResEncUNetLPlans"
CONFIG = "3d_fullres"
TRAINER_DIR = f"nnUNetTrainer__{PLANS_ID}__{CONFIG}"

nnUNet_raw = str(ROOT / "data" / "nnUNet_raw")
nnUNet_preprocessed = str(ROOT / "data" / "nnUNet_preprocessed")
nnUNet_results = str(ROOT / "results" / "nnUNet_results")

RESULTS_DIR = Path(nnUNet_results) / DATASET_NAME / TRAINER_DIR

ENV = {
    **os.environ,
    "PATH": f"{VENV_BIN}:{os.environ.get('PATH', '')}",
    "nnUNet_raw": nnUNet_raw,
    "nnUNet_preprocessed": nnUNet_preprocessed,
    "nnUNet_results": nnUNet_results,
    "nnUNet_compile": "0",
}

# ── Logging ────────────────────────────────────────────────────────────────────

def log(msg, level="INFO"):
    ts = datetime.now().strftime("%H:%M:%S")
    prefix = {"INFO": "✓", "WARN": "⚠", "ERROR": "✗", "STEP": "▶"}.get(level, "·")
    print(f"[{ts}] {prefix}  {msg}", flush=True)


def banner(msg):
    bar = "═" * min(70, len(msg) + 4)
    print(f"\n{bar}\n  {msg}\n{bar}", flush=True)


# ── Training log parsing ───────────────────────────────────────────────────────

def latest_log(fold: int) -> Path | None:
    pattern = str(RESULTS_DIR / f"fold_{fold}" / "training_log*.txt")
    files = sorted(glob.glob(pattern))
    return Path(files[-1]) if files else None


def parse_latest_metrics(fold: int) -> dict:
    """Return the most recently completed epoch's metrics."""
    lf = latest_log(fold)
    if not lf or not lf.exists():
        return {}

    epoch = train_loss = val_loss = dice = epoch_time = best_ema = None
    last_best = None

    with open(lf, "r", errors="replace") as f:
        for line in f:
            line = line.strip()
            m = re.search(r"Epoch (\d+)", line)
            if m:
                epoch = int(m.group(1))
                dice = None  # reset until we see Pseudo dice for this epoch
            m = re.search(r"train_loss\s+([-\d.e+]+)", line)
            if m: train_loss = float(m.group(1))
            m = re.search(r"val_loss\s+([-\d.e+]+)", line)
            if m: val_loss = float(m.group(1))
            m = re.search(r"Pseudo dice \[(.+?)\]", line)
            if m:
                inner = m.group(1)
                vals = re.findall(r"np\.float\d+\(([-\d.e+]+)\)", inner)
                if not vals:
                    vals = re.findall(r"(?<![a-zA-Z\d])(-?\d+\.\d+(?:[eE][+-]?\d+)?)", inner)
                if vals:
                    dice = [float(v) for v in vals]
            m = re.search(r"Epoch time:\s*([\d.]+)", line)
            if m: epoch_time = float(m.group(1))
            m = re.search(r"New best EMA pseudo Dice:\s*([\d.e+]+)", line)
            if m:
                best_ema = float(m.group(1))
                last_best = epoch

    return {
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "dice": dice,
        "mean_dice": (sum(dice) / len(dice)) if dice else None,
        "epoch_time": epoch_time,
        "best_ema": best_ema,
        "last_best_epoch": last_best,
    }


def is_saturated(fold: int, patience: int) -> tuple[bool, str]:
    """Return (True, reason) if training is saturated, else (False, '')."""
    m = parse_latest_metrics(fold)
    epoch = m.get("epoch")
    last_best = m.get("last_best_epoch")
    if epoch is None or last_best is None:
        return False, ""
    stale = epoch - last_best
    if stale >= patience:
        return True, f"No EMA improvement for {stale} epochs (last best: epoch {last_best})"
    return False, ""


# ── Post-processing ────────────────────────────────────────────────────────────

def generate_val_predictions(fold: int):
    """
    Run nnUNetv2_predict on the fold's validation split to create validation/
    predictions that post-processing can then clean.
    nnUNet only writes these automatically at epoch 1000 — we must generate them.
    """
    import json, os

    val_dir = RESULTS_DIR / f"fold_{fold}" / "validation"
    if val_dir.exists() and list(val_dir.glob("*.nii.gz")):
        log(f"Fold {fold}: validation predictions already exist ({len(list(val_dir.glob('*.nii.gz')))} files)")
        return True

    # Get val case IDs from the splits file
    splits_file = Path(nnUNet_preprocessed) / DATASET_NAME / "splits_final.json"
    if not splits_file.exists():
        log(f"No splits file at {splits_file}", "WARN")
        return False

    with open(splits_file) as f:
        splits = json.load(f)
    val_cases = splits[fold]["val"]

    # Symlink val images to a temp input dir
    raw_img = Path(nnUNet_raw) / DATASET_NAME / "imagesTr"
    tmp_in = Path(f"/tmp/pipeline_fold{fold}_val_input")
    tmp_in.mkdir(exist_ok=True)

    linked = 0
    for case in val_cases:
        src = raw_img / f"{case}_0000.nii.gz"
        dst = tmp_in / f"{case}_0000.nii.gz"
        if src.exists() and not dst.exists():
            os.symlink(src.resolve(), dst)
            linked += 1

    if linked == 0 and not list(tmp_in.glob("*.nii.gz")):
        log(f"No val images found in {raw_img}", "WARN")
        return False

    val_dir.mkdir(exist_ok=True)
    log(f"Fold {fold}: running nnUNetv2_predict on {len(val_cases)} val cases → {val_dir}", "STEP")

    ckpt = RESULTS_DIR / f"fold_{fold}" / "checkpoint_best.pth"
    cmd = [
        str(VENV_BIN / "nnUNetv2_predict"),
        "-i", str(tmp_in),
        "-o", str(val_dir),
        "-d", str(DATASET_ID),
        "-c", CONFIG,
        "-f", str(fold),
        "-tr", TRAINER,
        "-p", PLANS_ID,
        "--disable_tta",  # faster inference
    ]
    result = subprocess.run(cmd, env=ENV)
    if result.returncode == 0:
        log(f"Fold {fold}: val predictions done ({len(list(val_dir.glob('*.nii.gz')))} files)")
        return True
    else:
        log(f"Fold {fold}: nnUNetv2_predict failed (code {result.returncode})", "WARN")
        return False


def run_postprocessing(fold: int, strip_hw: float = 0.20, min_voxels: int = 50):
    """
    Generate val predictions (if not present) then post-process them.
    """
    # Step 1: Generate predictions if not already there
    log(f"Generating val predictions for fold {fold}…", "STEP")
    ok = generate_val_predictions(fold)
    if not ok:
        log(f"Fold {fold}: skipping post-processing (no val predictions)", "WARN")
        return

    val_dir = RESULTS_DIR / f"fold_{fold}" / "validation"
    pred_files = list(val_dir.glob("*.nii.gz"))
    if not pred_files:
        log(f"No .nii.gz files in {val_dir}", "WARN")
        return

    clean_dir = RESULTS_DIR / f"fold_{fold}" / "validation_postprocessed"
    clean_dir.mkdir(exist_ok=True)

    log(f"Post-processing {len(pred_files)} predictions → {clean_dir}")

    pp_script = ROOT / "scripts" / "postprocess_segmentation.py"
    cmd = [
        str(VENV_BIN / "python3"), str(pp_script),
        str(val_dir), "-o", str(clean_dir),
        "--strip-hw", str(strip_hw),
        "--min-voxels", str(min_voxels),
        "-v",
    ]
    result = subprocess.run(cmd, capture_output=False, env=ENV)
    if result.returncode == 0:
        log(f"Post-processing done → {clean_dir}")
    else:
        log(f"Post-processing returned code {result.returncode}", "WARN")


# ── Training subprocess ────────────────────────────────────────────────────────

def has_best_checkpoint(fold: int) -> bool:
    return (RESULTS_DIR / f"fold_{fold}" / "checkpoint_best.pth").exists()


def best_checkpoint_path(fold: int) -> Path | None:
    p = RESULTS_DIR / f"fold_{fold}" / "checkpoint_best.pth"
    return p if p.exists() else None


def train_fold(
    fold: int,
    *,
    max_epochs: int,
    patience: int,
    pretrained_weights: Path | None,
    strip_hw: float,
    min_voxels: int,
    resume: bool,
):
    banner(f"FOLD {fold} — max {max_epochs} epochs, patience {patience}")

    # Skip if already done and user requested resume
    if resume and has_best_checkpoint(fold):
        log(f"Fold {fold} already has checkpoint_best.pth — skipping training (--resume)", "WARN")
        run_postprocessing(fold, strip_hw, min_voxels)
        return

    # Build nnUNetv2_train command
    cmd = [
        str(VENV_BIN / "nnUNetv2_train"),
        str(DATASET_ID), CONFIG, str(fold),
        "-tr", TRAINER,
        "-p", PLANS_ID,
    ]
    if pretrained_weights and pretrained_weights.exists():
        cmd += ["-pretrained_weights", str(pretrained_weights)]
        log(f"Warm-starting from {pretrained_weights.name}")
    if resume or has_best_checkpoint(fold):
        cmd += ["--c"]   # continue from checkpoint_latest if exists
        log("Continuing from existing checkpoint (--c)")

    log(f"Launching: {' '.join(cmd)}", "STEP")
    # Use setsid so training + all its worker processes form a new group we can kill together
    proc = subprocess.Popen(cmd, env=ENV, start_new_session=True)

    poll_interval = 60   # seconds between saturation checks
    log_wait = 30        # seconds to wait before first log check

    log(f"Training started (PID {proc.pid}). Monitoring for saturation every {poll_interval}s…")
    time.sleep(log_wait)

    start_time = time.time()
    last_logged_epoch = -1

    try:
        while True:
            # Check if process already finished
            ret = proc.poll()
            if ret is not None:
                log(f"Training process exited with code {ret}")
                break

            # Read current metrics
            m = parse_latest_metrics(fold)
            epoch = m.get("epoch", 0) or 0

            # Log progress occasionally
            if epoch != last_logged_epoch and epoch > 0:
                mean_d = m.get("mean_dice")
                best = m.get("best_ema")
                stale = epoch - (m.get("last_best_epoch") or epoch)
                elapsed = timedelta(seconds=int(time.time() - start_time))
                et = m.get("epoch_time", 0) or 0
                eta_s = (max_epochs - epoch) * et
                eta = timedelta(seconds=int(eta_s))
                dice_str = (f"mean={mean_d:.4f}" if mean_d else "dice=n/a")
                log(
                    f"Fold {fold} | Epoch {epoch}/{max_epochs} | {dice_str} | "
                    f"bestEMA={best:.4f} | stale={stale}/{patience} | "
                    f"elapsed={elapsed} | ETA={eta}"
                )
                last_logged_epoch = epoch

            # Check epoch cap
            if epoch >= max_epochs:
                log(f"Fold {fold}: reached max_epochs={max_epochs}. Stopping.", "STEP")
                kill_process_group(proc)
                break

            # Early stopping check
            saturated, reason = is_saturated(fold, patience)
            if saturated:
                log(f"Fold {fold}: EARLY STOP — {reason}", "STEP")
                kill_process_group(proc)
                break

            time.sleep(poll_interval)

    except KeyboardInterrupt:
        log("Interrupted — terminating all training processes…", "WARN")
        kill_process_group(proc)
        sys.exit(1)

    # Summarize fold
    mean_d = m.get('mean_dice') or 0.0
    log(
        f"Fold {fold} complete | "
        f"epochs={m.get('epoch', '?')} | bestEMA={m.get('best_ema') or 0:.4f} | "
        f"lastMeanDice={float(mean_d):.4f}"
    )

    # Run post-processing of validation predictions
    log(f"Running post-processing on fold {fold} validation set…", "STEP")
    run_postprocessing(fold, strip_hw, min_voxels)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Integrated nnU-Net training pipeline: "
                    "fold-by-fold with early stopping and post-processing."
    )
    parser.add_argument("--start-fold", type=int, default=0,
                        help="First fold to train (default: 0)")
    parser.add_argument("--end-fold", type=int, default=4,
                        help="Last fold to train inclusive (default: 4)")
    parser.add_argument("--max-epochs", type=int, default=300,
                        help="Max epochs per fold (default: 300)")
    parser.add_argument("--patience", type=int, default=100,
                        help="Early-stop patience: stop after N epochs with no "
                             "EMA Dice improvement (default: 100)")
    parser.add_argument("--strip-hw", type=float, default=0.20,
                        help="Post-proc: sagittal strip half-width fraction (default: 0.20)")
    parser.add_argument("--min-voxels", type=int, default=50,
                        help="Post-proc: min component size (default: 50)")
    parser.add_argument("--pretrained-fold", type=int, default=0,
                        help="Use this fold's checkpoint_best as starting weights (default: 0)")
    parser.add_argument("--no-pretrained", action="store_true",
                        help="Do not use pretrained weights for any fold")
    parser.add_argument("--resume", action="store_true",
                        help="Skip folds that already have checkpoint_best.pth")
    args = parser.parse_args()

    banner("Spine Seg Pipeline — Early Stopping + Post-Processing")
    log(f"Folds: {args.start_fold}–{args.end_fold}")
    log(f"Max epochs: {args.max_epochs}  |  Patience: {args.patience} epochs")
    log(f"Post-proc strip-hw: {args.strip_hw}  |  min-voxels: {args.min_voxels}")

    # Get pretrained weights once (from --pretrained-fold best checkpoint)
    base_ckpt = best_checkpoint_path(args.pretrained_fold)
    if not args.no_pretrained and base_ckpt:
        log(f"Pretrained weights: fold_{args.pretrained_fold}/checkpoint_best.pth (bestEMA from that fold)")
    elif not args.no_pretrained:
        log(f"No checkpoint found for fold {args.pretrained_fold} — training from scratch", "WARN")

    for fold in range(args.start_fold, args.end_fold + 1):
        # For each fold after pretrained_fold, warm-start from either:
        # - The pretrained_fold checkpoint (first subsequent fold)
        # - The previous fold's best checkpoint (chain warm-starting)
        if args.no_pretrained:
            pretrained = None
        elif fold == args.pretrained_fold:
            pretrained = None  # this is the source fold, train normally
        else:
            # Try previous fold's best first, fall back to base
            prev = best_checkpoint_path(fold - 1)
            pretrained = prev if prev else base_ckpt

        train_fold(
            fold,
            max_epochs=args.max_epochs,
            patience=args.patience,
            pretrained_weights=pretrained,
            strip_hw=args.strip_hw,
            min_voxels=args.min_voxels,
            resume=args.resume,
        )

    banner("All folds complete! Running nnUNetv2_find_best_configuration…")
    cmd = [
        str(VENV_BIN / "nnUNetv2_find_best_configuration"),
        str(DATASET_ID), "-c", CONFIG, "-tr", TRAINER, "-p", PLANS_ID,
    ]
    subprocess.run(cmd, env=ENV)

    log("Pipeline complete. Run inference with:")
    log("  python scripts/run_inference.py --input <nifti_dir>")


if __name__ == "__main__":
    main()
