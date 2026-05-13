#!/usr/bin/env python3
"""
visualize_ensemble_flair.py
Generate fold-0-style 4-panel figures for ensemble FLAIR predictions:
  - Raw MRI (sagittal mid-slice)
  - Ground-truth segmentation overlay (if available)
  - Ensemble prediction overlay
  - Correct / Error map (if GT available) or Contour labels (if no GT)

For cases WITH ground truth (pseudo-labels from Dataset202):
  4 panels: Raw | GT | Prediction | Correct/Error + per-class Dice

For cases WITHOUT ground truth:
  3 panels: Raw | Prediction overlay | Contour labels + voxel counts

Outputs:
  results/ensemble_flair_visualizations_v2/<case>_panel.png
  results/ensemble_flair_visualizations_v2/summary_grid.png

Usage:
  python scripts/visualize_ensemble_flair.py [--n N]
"""

import argparse
import numpy as np
import nibabel as nib
import nibabel.orientations as nib_orient
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from pathlib import Path

BASE = Path(__file__).parent.parent

IMAGES_DIR = BASE / "data" / "flair_inference_input"
GT_DIR     = BASE / "data" / "nnUNet_raw" / "Dataset202_SpineL1L5_SSL" / "labelsTr"
PRED_DIR   = BASE / "results" / "ensemble_flair_raw"
OUT_DIR    = BASE / "results" / "ensemble_flair_visualizations_v2"

VERTEBRA_LABELS = [1, 2, 3, 4, 5]
LABEL_NAMES = ["BG", "L1", "L2", "L3", "L4", "L5"]
COLORS_HEX  = ["none", "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4"]
COLORS_RGB  = [(230, 25, 75), (60, 180, 75), (67, 99, 216), (245, 130, 49), (145, 30, 180)]
SEG_CMAP    = ListedColormap(["none"] + COLORS_HEX[1:])
ALPHA       = 0.45


def load_volume_ras(path):
    """Load NIfTI and reorient to RAS+ for consistent viewing."""
    img = nib.load(str(path))
    data = img.get_fdata()
    ornt = nib_orient.io_orientation(img.affine)
    target = nib_orient.axcodes2ornt(('R', 'A', 'S'))
    xfm = nib_orient.ornt_transform(ornt, target)
    return nib_orient.apply_orientation(data, xfm)


def normalise(arr):
    lo, hi = np.percentile(arr, 1), np.percentile(arr, 99)
    return np.clip((arr - lo) / (hi - lo + 1e-8), 0, 1)


def best_slice_idx(seg_vol):
    """Return index of slice (axis 0) with most foreground voxels."""
    fg = (seg_vol > 0).reshape(seg_vol.shape[0], -1).sum(axis=1)
    if fg.max() == 0:
        return seg_vol.shape[0] // 2
    return int(np.argmax(fg))


def overlay(ax, img2d, seg2d, cmap, title, title_color="white"):
    ax.imshow(img2d, cmap="gray", origin="lower", aspect="auto")
    masked = np.ma.masked_where(seg2d == 0, seg2d.astype(float))
    ax.imshow(masked, cmap=cmap, origin="lower", aspect="auto",
              alpha=ALPHA, vmin=0, vmax=5)
    ax.set_title(title, fontsize=10, fontweight="bold", color=title_color, pad=4)
    ax.axis("off")


def dice(gt2d, pred2d, cls):
    tp    = ((gt2d == cls) & (pred2d == cls)).sum()
    denom = (gt2d == cls).sum() + (pred2d == cls).sum()
    return 2 * tp / denom if denom > 0 else float("nan")


def make_panel_with_gt(case_id, img_path, gt_path, pred_path, out_path):
    """4-panel: Raw MRI | Ground Truth | Prediction | Correct/Error with Dice."""
    img_vol  = load_volume_ras(img_path)
    gt_vol   = load_volume_ras(gt_path)
    pred_vol = load_volume_ras(pred_path)

    # Ensure matching shapes
    min_shape = tuple(min(a, b, c) for a, b, c in zip(img_vol.shape, gt_vol.shape, pred_vol.shape))
    img_vol  = img_vol[:min_shape[0], :min_shape[1], :min_shape[2]]
    gt_vol   = gt_vol[:min_shape[0], :min_shape[1], :min_shape[2]]
    pred_vol = pred_vol[:min_shape[0], :min_shape[1], :min_shape[2]]

    idx    = best_slice_idx(gt_vol)
    img2d  = normalise(img_vol[idx])
    gt2d   = gt_vol[idx].astype(int)
    pred2d = pred_vol[idx].astype(int)

    # Diff: 1=correct foreground, 2=wrong class
    diff2d = np.zeros_like(gt2d)
    diff2d[(gt2d > 0) & (pred2d == gt2d)] = 1
    diff2d[(gt2d > 0) & (pred2d != gt2d)] = 2

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.patch.set_facecolor("#111111")
    for ax in axes:
        ax.set_facecolor("#111111")

    # Panel 1: Raw MRI
    axes[0].imshow(img2d, cmap="gray", origin="lower", aspect="auto")
    axes[0].set_title("Raw MRI", fontsize=10, fontweight="bold", color="white", pad=4)
    axes[0].axis("off")

    # Panel 2: Ground Truth
    overlay(axes[1], img2d, gt2d, SEG_CMAP, "Ground Truth")

    # Panel 3: Prediction
    overlay(axes[2], img2d, pred2d, SEG_CMAP, "Prediction")

    # Panel 4: Correct vs Error
    diff_cmap = ListedColormap(["#111111", "#00cc66", "#ff3333"])
    axes[3].imshow(img2d, cmap="gray", origin="lower", aspect="auto")
    axes[3].imshow(np.ma.masked_where(diff2d == 0, diff2d),
                   cmap=diff_cmap, origin="lower", aspect="auto",
                   alpha=0.6, vmin=0, vmax=2)
    axes[3].set_title("Correct / Error", fontsize=10, fontweight="bold", color="white", pad=4)
    axes[3].axis("off")

    # Legend
    patches = [mpatches.Patch(color=c, label=n)
               for c, n in zip(COLORS_HEX[1:], LABEL_NAMES[1:])]
    fig.legend(handles=patches, loc="lower center", ncol=5, fontsize=9,
               facecolor="#222222", edgecolor="none", labelcolor="white",
               bbox_to_anchor=(0.5, -0.02))

    # Per-class Dice in title
    dice_str = [f"{LABEL_NAMES[c]}: {dice(gt2d, pred2d, c):.2f}" for c in range(1, 6)]
    fig.suptitle(f"{case_id}   |   Dice — " + "  ".join(dice_str),
                 fontsize=9, color="white", y=1.01)

    plt.tight_layout(pad=0.5)
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight", facecolor="#111111")
    plt.close(fig)

    # Return vertebrae count
    present = {LABEL_NAMES[l]: int((pred_vol == l).sum())
               for l in VERTEBRA_LABELS if (pred_vol == l).any()}
    return True, present


def make_panel_no_gt(case_id, img_path, pred_path, out_path):
    """3-panel: Raw MRI | Prediction overlay | Contour labels (no GT available)."""
    import cv2

    img_vol  = load_volume_ras(img_path)
    pred_vol = load_volume_ras(pred_path)

    # Ensure matching shapes
    min_shape = tuple(min(a, b) for a, b in zip(img_vol.shape, pred_vol.shape))
    img_vol  = img_vol[:min_shape[0], :min_shape[1], :min_shape[2]]
    pred_vol = pred_vol[:min_shape[0], :min_shape[1], :min_shape[2]]

    idx    = best_slice_idx(pred_vol)
    img2d  = normalise(img_vol[idx])
    pred2d = pred_vol[idx].astype(int)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.patch.set_facecolor("#111111")
    for ax in axes:
        ax.set_facecolor("#111111")

    # Panel 1: Raw MRI
    axes[0].imshow(img2d, cmap="gray", origin="lower", aspect="auto")
    axes[0].set_title("Raw FLAIR MRI", fontsize=10, fontweight="bold", color="white", pad=4)
    axes[0].axis("off")

    # Panel 2: Prediction overlay
    overlay(axes[1], img2d, pred2d, SEG_CMAP, "Ensemble Prediction")

    # Panel 3: Contour overlay with labels
    norm_u8 = (img2d * 255).astype(np.uint8)
    overlay_bgr = cv2.cvtColor(norm_u8, cv2.COLOR_GRAY2BGR)

    for lab_val, color in zip(VERTEBRA_LABELS, COLORS_RGB):
        vert_mask = (pred2d == lab_val).astype(np.uint8)
        if vert_mask.sum() == 0:
            continue
        color_mask = np.zeros_like(overlay_bgr)
        color_mask[vert_mask > 0] = color[::-1]
        overlay_bgr = cv2.addWeighted(overlay_bgr, 1.0, color_mask, 0.35, 0)
        contours, _ = cv2.findContours(vert_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay_bgr, contours, -1, color[::-1], 2)
        coords = np.where(vert_mask > 0)
        if len(coords[0]) > 0:
            cy, cx = int(np.mean(coords[0])), int(np.mean(coords[1]))
            name = LABEL_NAMES[lab_val]
            cv2.putText(overlay_bgr, name, (cx - 12, cy + 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
            cv2.putText(overlay_bgr, name, (cx - 12, cy + 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color[::-1], 2)

    overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)
    axes[2].imshow(overlay_rgb, origin="lower", aspect="auto")
    axes[2].set_title("Vertebra Labels", fontsize=10, fontweight="bold", color="white", pad=4)
    axes[2].axis("off")

    # Legend
    patches = [mpatches.Patch(color=c, label=n)
               for c, n in zip(COLORS_HEX[1:], LABEL_NAMES[1:])]
    fig.legend(handles=patches, loc="lower center", ncol=5, fontsize=9,
               facecolor="#222222", edgecolor="none", labelcolor="white",
               bbox_to_anchor=(0.5, -0.02))

    # Title with voxel counts
    present = {LABEL_NAMES[l]: int((pred_vol == l).sum())
               for l in VERTEBRA_LABELS if (pred_vol == l).any()}
    det_str = ", ".join(f"{k}: {v:,} vox" for k, v in present.items()) if present else "No vertebrae detected"
    fig.suptitle(f"{case_id}   |   Detected: {det_str}",
                 fontsize=9, color="white", y=1.01)

    plt.tight_layout(pad=0.5)
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight", facecolor="#111111")
    plt.close(fig)

    return False, present


def make_summary_grid(panel_paths, out_path):
    """Combine individual panels into a summary grid image."""
    imgs = [plt.imread(str(p)) for p in panel_paths]
    cols = min(2, len(imgs))
    rows = (len(imgs) + cols - 1) // cols
    h, w = imgs[0].shape[:2]
    grid = np.ones((rows * h, cols * w, 3), dtype=np.float32)
    for i, im in enumerate(imgs):
        r, c = divmod(i, cols)
        if im.ndim == 2:
            im = np.stack([im] * 3, axis=-1)
        if im.shape[2] == 4:
            im = im[..., :3]
        # Handle size mismatches
        ih, iw = im.shape[:2]
        ph, pw = min(ih, h), min(iw, w)
        grid[r*h:r*h+ph, c*w:c*w+pw] = im[:ph, :pw]

    fig, ax = plt.subplots(figsize=(cols * 10, rows * 4))
    fig.patch.set_facecolor("#111111")
    ax.imshow(grid)
    ax.axis("off")
    plt.tight_layout(pad=0)
    fig.savefig(str(out_path), dpi=120, bbox_inches="tight", facecolor="#111111")
    plt.close(fig)
    print(f"\n  Summary grid -> {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate fold-0-style visualizations for ensemble FLAIR predictions")
    parser.add_argument("--n", type=int, default=0,
                        help="Max cases to visualize (0=all)")
    parser.add_argument("--force", action="store_true",
                        help="Regenerate existing panels")
    parser.add_argument("--summary-n", type=int, default=20,
                        help="Number of panels to include in summary grid")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Find all predictions
    pred_files = sorted(PRED_DIR.glob("*.nii.gz"))
    if args.n > 0:
        pred_files = pred_files[:args.n]

    print(f"\n{'='*70}")
    print(f"Ensemble FLAIR Visualization (fold-0 style)")
    print(f"  Predictions : {PRED_DIR} ({len(pred_files)} files)")
    print(f"  Images      : {IMAGES_DIR}")
    print(f"  Ground Truth: {GT_DIR}")
    print(f"  Output      : {OUT_DIR}")
    print(f"{'='*70}\n")

    panel_paths = []
    n_with_gt = 0
    n_no_gt = 0
    n_skipped = 0
    n_error = 0

    for i, pred_file in enumerate(pred_files):
        case_id = pred_file.name.replace(".nii.gz", "")
        out_png = OUT_DIR / f"{case_id}_panel.png"

        if out_png.exists() and not args.force:
            panel_paths.append(out_png)
            n_skipped += 1
            continue

        # Find matching input image
        img_file = IMAGES_DIR / f"{case_id}_0000.nii.gz"
        if not img_file.exists():
            continue

        # Check for ground truth
        gt_file = GT_DIR / f"{case_id}.nii.gz"
        has_gt = gt_file.exists()

        try:
            if has_gt:
                had_gt, present = make_panel_with_gt(case_id, img_file, gt_file, pred_file, out_png)
                n_with_gt += 1
                n_vert = len(present)
                status = f"{n_vert}v + GT"
            else:
                had_gt, present = make_panel_no_gt(case_id, img_file, pred_file, out_png)
                n_no_gt += 1
                n_vert = len(present)
                status = f"{n_vert}v"

            panel_paths.append(out_png)
            if (i + 1) % 20 == 0 or i < 5:
                print(f"  [{i+1}/{len(pred_files)}] {case_id} -- {status}")

        except Exception as e:
            n_error += 1
            print(f"  ERROR {case_id}: {e}")

    # Summary grid (first N with GT panels, then no-GT)
    if panel_paths:
        grid_panels = panel_paths[:args.summary_n]
        grid_path = OUT_DIR / "summary_grid.png"
        make_summary_grid(grid_panels, grid_path)

    print(f"\n{'='*70}")
    print(f"Visualization complete!")
    print(f"  With GT (4-panel) : {n_with_gt}")
    print(f"  No GT (3-panel)   : {n_no_gt}")
    print(f"  Skipped (exist)   : {n_skipped}")
    print(f"  Errors            : {n_error}")
    print(f"  Total panels      : {len(panel_paths)}")
    print(f"  Output dir        : {OUT_DIR}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
