#!/usr/bin/env python3
"""
Live nnU-Net training monitor — spine L1-L5 segmentation.
Shows per-epoch: train/val loss, pseudo Dice per vertebra, LR, epoch time, ETA.

Usage:
    python scripts/monitor_training.py
    python scripts/monitor_training.py --fold 0          # watch specific fold
    python scripts/monitor_training.py --refresh 5       # refresh interval (seconds)
"""

import os
import re
import sys
import time
import glob
import argparse
import shutil
from pathlib import Path
from datetime import datetime, timedelta


# ─── ANSI colors ────────────────────────────────────────────────────────────
R = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
MAGENTA = "\033[95m"
BLUE = "\033[94m"
WHITE = "\033[97m"
BG_DARK = "\033[40m"

LABELS = ["L1", "L2", "L3", "L4", "L5"]
TOTAL_EPOCHS = 1000


def clear():
    os.system("clear" if os.name == "posix" else "cls")


def term_width():
    return shutil.get_terminal_size((120, 40)).columns


def bar(value, max_val=1.0, width=20, filled="█", empty="░",
        color=GREEN, warn=0.5, good=0.7):
    """Render a coloured progress bar."""
    pct = min(1.0, max(0.0, value / max_val)) if max_val > 0 else 0
    n = int(pct * width)
    if value >= good:
        c = GREEN
    elif value >= warn:
        c = YELLOW
    else:
        c = RED
    return f"{c}{filled * n}{DIM}{empty * (width - n)}{R}"


def loss_bar(value, width=20):
    """Loss bar: high loss = red, low = green (inverted)."""
    pct = min(1.0, max(0.0, value))
    n = int(pct * width)
    c = RED if value > 0.5 else (YELLOW if value > 0.2 else GREEN)
    return f"{c}{'█' * n}{DIM}{'░' * (width - n)}{R}"


def find_log_files(results_dir, fold):
    """Find the latest training log for the given fold."""
    pattern = str(results_dir / f"fold_{fold}" / "training_log*.txt")
    files = sorted(glob.glob(pattern))
    return files[-1] if files else None


def parse_log(log_path):
    """Parse nnUNet training log, return list of epoch dicts."""
    epochs = []
    current = {}
    best_ema = None

    with open(log_path, "r", errors="replace") as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()

        m = re.search(r"Epoch (\d+)", line)
        if m:
            if current and "epoch" in current:
                epochs.append(current)
            current = {"epoch": int(m.group(1))}

        m = re.search(r"Current learning rate:\s*([\d.e+-]+)", line)
        if m:
            current["lr"] = float(m.group(1))

        m = re.search(r"train_loss\s+([\d.e+-]+)", line)
        if m:
            current["train_loss"] = float(m.group(1))

        m = re.search(r"val_loss\s+([\d.e+-]+)", line)
        if m:
            current["val_loss"] = float(m.group(1))

        m = re.search(r"Pseudo dice \[(.+?)\]", line)
        if m:
            inner = m.group(1)
            # Handle both plain floats and np.float32(val) format
            np_vals = re.findall(r"np\.float\d+\(([-\d.e+]+)\)", inner)
            if np_vals:
                current["dice"] = [float(v) for v in np_vals]
            else:
                plain_vals = re.findall(r"(?<![a-zA-Z\d])(-?\d+\.\d+(?:[eE][+-]?\d+)?)", inner)
                current["dice"] = [float(v) for v in plain_vals if v]

        m = re.search(r"Epoch time:\s*([\d.]+)", line)
        if m:
            current["epoch_time"] = float(m.group(1))

        m = re.search(r"New best EMA pseudo Dice:\s*([\d.e+-]+)", line)
        if m:
            best_ema = float(m.group(1))
            current["best_ema"] = best_ema

        m = re.search(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", line)
        if m:
            current["last_ts"] = m.group(1)

    if current and "epoch" in current:
        epochs.append(current)

    return epochs, best_ema


def find_all_folds(results_dir):
    folds = []
    for d in sorted(results_dir.glob("fold_*")):
        fold_num = int(d.name.split("_")[1])
        folds.append(fold_num)
    return folds


def render_epoch_table(epochs, max_rows=20, w=120):
    """Render the per-epoch metric table."""
    if not epochs:
        return [f"  {DIM}No epochs yet...{R}"]

    lines = []
    sep = "─"

    header = (
        f"  {'Epoch':>5}  {'Train Loss':>10}  {'Val Loss':>8}  "
        f"{'L1':>6} {'L2':>6} {'L3':>6} {'L4':>6} {'L5':>6}  "
        f"{'Mean Dice':>9}  {'LR':>8}  {'Time':>7}"
    )
    lines.append(f"{BOLD}{WHITE}{header}{R}")
    lines.append(f"  {DIM}{sep * (len(header) - 2)}{R}")

    # Show last max_rows epochs
    show = epochs[-max_rows:]
    for ep in show:
        e = ep.get("epoch", "?")
        tl = ep.get("train_loss", float("nan"))
        vl = ep.get("val_loss", float("nan"))
        dice = ep.get("dice", [])
        lr = ep.get("lr", float("nan"))
        et = ep.get("epoch_time", 0)
        is_best = "best_ema" in ep

        # Dice values
        dl = [f"{d:6.4f}" if d > 0 else f"{DIM}  0.00 {R}" for d in dice]
        dl += ["  n/a "] * (5 - len(dl))

        mean_d = sum(dice) / len(dice) if dice else 0.0
        dice_str = " ".join(dl[:5])

        tl_c = GREEN if tl < 0.2 else (YELLOW if tl < 0.5 else RED)
        vl_c = GREEN if vl < 0.15 else (YELLOW if vl < 0.4 else RED)
        star = f" {YELLOW}★{R}" if is_best else "  "
        md_c = GREEN if mean_d > 0.7 else (YELLOW if mean_d > 0.4 else DIM)

        row = (
            f"  {CYAN}{e:>5}{R}  "
            f"{tl_c}{tl:>10.4f}{R}  "
            f"{vl_c}{vl:>8.4f}{R}  "
            f"{DIM}{dice_str}{R}  "
            f"{md_c}{mean_d:>9.4f}{R}  "
            f"{DIM}{lr:>8.6f}{R}  "
            f"{DIM}{et:>6.1f}s{R}"
            f"{star}"
        )
        lines.append(row)

    return lines


def render_dice_sparkline(epochs, label_idx, width=40):
    """ASCII sparkline for dice over time for one label."""
    vals = [ep.get("dice", [0] * 5)[label_idx] if len(ep.get("dice", [])) > label_idx else 0
            for ep in epochs]
    if not vals:
        return "─" * width
    chars = " ▁▂▃▄▅▆▇█"
    vmin, vmax = 0.0, 1.0
    out = ""
    step = max(1, len(vals) // width)
    sampled = vals[::step][-width:]
    for v in sampled:
        idx = int((v - vmin) / (vmax - vmin) * (len(chars) - 1))
        c = GREEN if v > 0.7 else (YELLOW if v > 0.4 else DIM)
        out += f"{c}{chars[max(0, min(idx, len(chars)-1))]}{R}"
    return out.ljust(width)


def render_dashboard(args, results_dir):
    w = term_width()
    available_folds = find_all_folds(results_dir)
    folds_to_show = [args.fold] if args.fold is not None else available_folds

    lines = []
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Header
    title = "  nnU-Net Training Monitor — Spine L1-L5 Segmentation (RTX 5090)"
    lines.append(f"\n{BOLD}{CYAN}{'═' * min(w-1, 100)}{R}")
    lines.append(f"{BOLD}{WHITE}{title}{R}")
    lines.append(f"  {DIM}Updated: {now}  |  Dataset: Dataset100_SpineL1L5  |  Arch: ResEncUNet-L 3d_fullres  |  Refresh: {args.refresh}s{R}")
    lines.append(f"{BOLD}{CYAN}{'═' * min(w-1, 100)}{R}\n")

    if not available_folds:
        lines.append(f"  {YELLOW}Waiting for training to start...{R}")
        lines.append(f"  {DIM}Looking in: {results_dir}{R}")
        return "\n".join(lines)

    all_fold_data = {}
    for fold in folds_to_show:
        log = find_log_files(results_dir, fold)
        if log:
            epochs, best_ema = parse_log(log)
            all_fold_data[fold] = (epochs, best_ema, log)

    # Summary row for all folds
    lines.append(f"{BOLD}  Fold Summary{R}")
    lines.append(f"  {'─' * 60}")
    for fold in sorted(all_fold_data.keys()):
        epochs, best_ema, log = all_fold_data[fold]
        n = len(epochs)
        pct = n / TOTAL_EPOCHS * 100
        last = epochs[-1] if epochs else {}
        last_dice = last.get("dice", [])
        mean_d = sum(last_dice) / len(last_dice) if last_dice else 0
        best_str = f"{best_ema:.4f}" if best_ema is not None else "n/a"
        eta_s = ((TOTAL_EPOCHS - n) * last.get("epoch_time", 120)) if n > 0 else 0
        eta = str(timedelta(seconds=int(eta_s)))

        status_c = GREEN if n >= TOTAL_EPOCHS else (CYAN if n > 0 else DIM)
        status = "DONE" if n >= TOTAL_EPOCHS else ("ACTIVE" if n > 0 else "WAITING")

        prog_bar = bar(n, TOTAL_EPOCHS, width=30)
        lines.append(
            f"  {BOLD}Fold {fold}{R}  [{prog_bar}] "
            f"{status_c}{n:>4}/{TOTAL_EPOCHS}{R} ({pct:4.1f}%)  "
            f"Best EMA Dice: {GREEN if best_ema and best_ema > 0.7 else YELLOW}{best_str}{R}  "
            f"Mean Dice: {GREEN if mean_d > 0.7 else YELLOW}{mean_d:.4f}{R}  "
            f"ETA: {DIM}{eta}{R}  "
            f"[{status_c}{status}{R}]"
        )
    lines.append("")

    # Detailed view for each fold
    for fold in sorted(all_fold_data.keys()):
        epochs, best_ema, log = all_fold_data[fold]
        if not epochs:
            continue

        last = epochs[-1]
        last_dice = last.get("dice", [0] * 5)
        mean_d = sum(last_dice) / len(last_dice) if last_dice else 0

        lines.append(f"{BOLD}{MAGENTA}  ── Fold {fold} ─────────────────────────────────────────────────{R}")

        # Stats row
        tl = last.get("train_loss", 0)
        vl = last.get("val_loss", 0)
        lr = last.get("lr", 0)
        et = last.get("epoch_time", 0)
        ts = last.get("last_ts", "")
        lines.append(
            f"  Last epoch: {CYAN}{last.get('epoch', '?')}{R}  "
            f"Train Loss: {RED if tl>0.3 else GREEN}{tl:.4f}{R}  "
            f"Val Loss: {RED if vl>0.2 else GREEN}{vl:.4f}{R}  "
            f"LR: {DIM}{lr:.6f}{R}  "
            f"Epoch time: {DIM}{et:.1f}s{R}  "
            f"Best EMA: {GREEN}{best_ema:.4f}{R}" if best_ema else
            f"  Last epoch: {CYAN}{last.get('epoch', '?')}{R}"
        )

        # Per-vertebra dice bars
        lines.append(f"\n  {BOLD}Per-Vertebra Pseudo Dice (last epoch):{R}")
        for i, label in enumerate(LABELS):
            d = last_dice[i] if i < len(last_dice) else 0
            b = bar(d, 1.0, width=30)
            trend = ""
            if len(epochs) >= 5:
                prev = epochs[-5].get("dice", [0]*5)
                delta = d - (prev[i] if i < len(prev) else 0)
                trend = f" {GREEN}↑{R}" if delta > 0.001 else (f" {RED}↓{R}" if delta < -0.001 else f" {DIM}─{R}")
            lines.append(f"    {BOLD}{label}{R}  {b}  {GREEN if d>0.7 else YELLOW if d>0.4 else RED}{d:.4f}{R}{trend}")

        # Sparklines
        lines.append(f"\n  {BOLD}Dice History (all epochs, last epoch rightmost):{R}")
        for i, label in enumerate(LABELS):
            spark = render_dice_sparkline(epochs, i, width=min(60, w - 20))
            lines.append(f"    {BOLD}{label}{R}  {spark}")

        # Loss trend
        lines.append(f"\n  {BOLD}Loss History:{R}")
        train_vals = [ep.get("train_loss", 0) for ep in epochs]
        val_vals = [ep.get("val_loss", 0) for ep in epochs]
        chars = " ▁▂▃▄▅▆▇█"
        spark_w = min(60, w - 20)
        step = max(1, len(train_vals) // spark_w)

        t_spark = ""
        for v in train_vals[::step][-spark_w:]:
            idx = int(min(v, 1.0) * (len(chars) - 1))
            t_spark += f"{RED}{chars[idx]}{R}"
        v_spark = ""
        for v in val_vals[::step][-spark_w:]:
            idx = int(min(v, 1.0) * (len(chars) - 1))
            v_spark += f"{BLUE}{chars[idx]}{R}"

        lines.append(f"    {RED}Train{R}  {t_spark}")
        lines.append(f"    {BLUE}Val  {R}  {v_spark}")

        # Epoch table (last 15 epochs)
        lines.append(f"\n  {BOLD}Recent Epochs:{R}")
        table = render_epoch_table(epochs, max_rows=15, w=w)
        lines.extend(["  " + l for l in table])
        lines.append("")

    lines.append(f"{DIM}  Log: {log}{R}")
    lines.append(f"  {DIM}Press Ctrl+C to exit{R}\n")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Live nnU-Net training monitor")
    parser.add_argument("--results", default=None,
                        help="Path to nnUNet_results dir (auto-detected if omitted)")
    parser.add_argument("--fold", type=int, default=None,
                        help="Specific fold to monitor (default: all)")
    parser.add_argument("--refresh", type=float, default=10.0,
                        help="Refresh interval in seconds (default: 10)")
    parser.add_argument("--once", action="store_true",
                        help="Print once and exit (no live refresh)")
    args = parser.parse_args()

    # Auto-detect results dir
    if args.results:
        results_root = Path(args.results)
    else:
        # Walk up to find results/nnUNet_results
        script_dir = Path(__file__).parent.parent
        candidates = [
            script_dir / "results" / "nnUNet_results",
            Path(os.environ.get("nnUNet_results", "/nonexistent")),
        ]
        results_root = None
        for c in candidates:
            if c.exists():
                results_root = c
                break
        if not results_root:
            print(f"{RED}Could not find nnUNet_results directory.{R}")
            print(f"Set --results or export nnUNet_results env var.")
            sys.exit(1)

    # Find dataset dir
    ds_dirs = list(results_root.glob("Dataset*_SpineL1L5"))
    if not ds_dirs:
        ds_dirs = list(results_root.glob("Dataset*"))
    if not ds_dirs:
        print(f"{YELLOW}Waiting for training to start in {results_root}...{R}")
        if args.once:
            return
        time.sleep(args.refresh)
        return main()

    # Find trainer dir
    trainer_dirs = []
    for ds in ds_dirs:
        trainer_dirs += list(ds.glob("nnUNetTrainer*"))
    if not trainer_dirs:
        trainer_dirs = ds_dirs

    results_dir = trainer_dirs[0]

    if args.once:
        print(render_dashboard(args, results_dir))
        return

    try:
        while True:
            clear()
            print(render_dashboard(args, results_dir))
            time.sleep(args.refresh)
    except KeyboardInterrupt:
        print(f"\n{DIM}Monitor stopped.{R}\n")


if __name__ == "__main__":
    main()
