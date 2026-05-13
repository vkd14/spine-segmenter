"""
visualize_ensemble_4panel.py
Generate presentation-quality 4-panel figures using 5-fold ensemble predictions.
Matches the exact layout of the fold 0 visualizations (Raw MRI | GT | Prediction | Correct/Error).

Uses nibabel's affine-based reorientation (RAS+) for correct spatial alignment.

Outputs:
  results/ensemble_visualizations/<case>_panel.png
  results/ensemble_visualizations/summary_grid.png

Usage:
  python scripts/visualize_ensemble_4panel.py [--n N] [--pred-dir PATH]
"""

import argparse
import numpy as np
import nibabel as nib
import nibabel.orientations as nib_orient
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from pathlib import Path

BASE       = Path(__file__).resolve().parent.parent
IMAGES_DIR = BASE / "data" / "nnUNet_raw" / "Dataset202_SpineL1L5_SSL" / "imagesTr"
GT_DIR     = BASE / "data" / "nnUNet_raw" / "Dataset202_SpineL1L5_SSL" / "labelsTr"
# Default: use the postprocessed ensemble predictions
DEFAULT_PRED_DIR = BASE / "results" / "ensemble_flair_postprocessed"
DEFAULT_OUT_DIR  = BASE / "results" / "ensemble_visualizations"

LABEL_NAMES = ["BG", "L1", "L2", "L3", "L4", "L5"]
COLORS      = ["none", "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4"]
SEG_CMAP    = ListedColormap(["none"] + COLORS[1:])
ALPHA       = 0.45

def load_volume(path):
    """Load NIfTI and reorient to RAS+ for consistent axis alignment."""
    img = nib.load(str(path))
    data = img.get_fdata()
    ornt = nib_orient.io_orientation(img.affine)
    target_ornt = nib_orient.axcodes2ornt(('R', 'A', 'S'))
    transform = nib_orient.ornt_transform(ornt, target_ornt)
    return nib_orient.apply_orientation(data, transform)

def normalise(arr):
    lo, hi = np.percentile(arr, 1), np.percentile(arr, 99)
    return np.clip((arr - lo) / (hi - lo + 1e-8), 0, 1)

def best_slice_idx(seg_vol):
    """Return index of slice (axis 0) with most foreground voxels."""
    fg = (seg_vol > 0).reshape(seg_vol.shape[0], -1).sum(axis=1)
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

def make_panel(case_id, img_path, gt_path, pred_path, out_path):
    img_vol  = load_volume(img_path)
    gt_vol   = load_volume(gt_path)
    pred_vol = load_volume(pred_path)

    # Ensure matching shapes by trimming to common dimensions
    min_shape = tuple(min(a, b, c) for a, b, c in zip(img_vol.shape, gt_vol.shape, pred_vol.shape))
    img_vol  = img_vol[:min_shape[0], :min_shape[1], :min_shape[2]]
    gt_vol   = gt_vol[:min_shape[0], :min_shape[1], :min_shape[2]]
    pred_vol = pred_vol[:min_shape[0], :min_shape[1], :min_shape[2]]

    # Find best slice based on GT (or pred if GT is empty)
    gt_fg = (gt_vol > 0).reshape(gt_vol.shape[0], -1).sum(axis=1)
    pred_fg = (pred_vol > 0).reshape(pred_vol.shape[0], -1).sum(axis=1)
    
    if gt_fg.max() > 0:
        idx = int(np.argmax(gt_fg))
    elif pred_fg.max() > 0:
        idx = int(np.argmax(pred_fg))
    else:
        idx = gt_vol.shape[0] // 2

    img2d   = normalise(img_vol[idx])
    gt2d    = gt_vol[idx].astype(int)
    pred2d  = pred_vol[idx].astype(int)

    # Diff: 1=correct foreground, 2=wrong class or missed
    diff2d = np.zeros_like(gt2d)
    diff2d[(gt2d > 0) & (pred2d == gt2d)] = 1
    diff2d[(gt2d > 0) & (pred2d != gt2d)] = 2

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.patch.set_facecolor("#111111")
    for ax in axes: ax.set_facecolor("#111111")

    # Panel 1: raw MRI
    axes[0].imshow(img2d, cmap="gray", origin="lower", aspect="auto")
    axes[0].set_title("Raw MRI", fontsize=10, fontweight="bold", color="white", pad=4)
    axes[0].axis("off")

    # Panel 2: GT
    overlay(axes[1], img2d, gt2d, SEG_CMAP, "Ground Truth")

    # Panel 3: Ensemble Prediction
    overlay(axes[2], img2d, pred2d, SEG_CMAP, "Ensemble Prediction")

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
               for c, n in zip(COLORS[1:], LABEL_NAMES[1:])]
    fig.legend(handles=patches, loc="lower center", ncol=5, fontsize=9,
               facecolor="#222222", edgecolor="none", labelcolor="white",
               bbox_to_anchor=(0.5, -0.02))

    # Per-class Dice in title
    dice_vals = [dice(gt2d, pred2d, c) for c in range(1, 6)]
    dice_str = [f"{LABEL_NAMES[c]}: {dice(gt2d, pred2d, c):.2f}" for c in range(1, 6)]
    mean_dice = np.nanmean(dice_vals)
    fig.suptitle(f"{case_id}   |   Dice — " + "  ".join(dice_str) + f"  |  Mean: {mean_dice:.2f}",
                 fontsize=9, color="white", y=1.01)

    plt.tight_layout(pad=0.5)
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight",
                facecolor="#111111")
    plt.close(fig)
    print(f"  ✓ {out_path.name}  (mean Dice: {mean_dice:.3f})")
    return dice_str, mean_dice

def make_summary_grid(panel_paths, out_dir):
    imgs = [plt.imread(str(p)) for p in panel_paths]
    cols = min(2, len(imgs))
    rows = (len(imgs) + cols - 1) // cols
    h, w = imgs[0].shape[:2]
    grid = np.ones((rows * h, cols * w, 3), dtype=np.float32)
    for i, im in enumerate(imgs):
        r, c = divmod(i, cols)
        if im.ndim == 2:
            im = np.stack([im]*3, axis=-1)
        if im.shape[2] == 4:
            im = im[..., :3]
        grid[r*h:(r+1)*h, c*w:(c+1)*w] = im[:h, :w]
    out = out_dir / "summary_grid.png"
    fig, ax = plt.subplots(figsize=(cols * 10, rows * 4))
    fig.patch.set_facecolor("#111111")
    ax.imshow(grid); ax.axis("off")
    plt.tight_layout(pad=0)
    fig.savefig(str(out), dpi=120, bbox_inches="tight", facecolor="#111111")
    plt.close(fig)
    print(f"\n✓ Summary grid → {out}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=0,
                        help="Number of cases to visualize (0 = all)")
    parser.add_argument("--pred-dir", type=str, default=str(DEFAULT_PRED_DIR),
                        help="Directory with ensemble predictions")
    parser.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR),
                        help="Output directory for visualizations")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_dir = Path(args.pred_dir)

    # Find matching cases: need image + GT + prediction
    pred_stems = {p.name.replace(".nii.gz","") for p in pred_dir.glob("*.nii.gz")}
    img_stems  = {p.name.replace("_0000.nii.gz","") for p in IMAGES_DIR.glob("*.nii.gz")}
    gt_stems   = {p.name.replace(".nii.gz","") for p in GT_DIR.glob("*.nii.gz")}
    
    # Visualize cases where we have ALL three: image, GT, and prediction
    matched = sorted(pred_stems & img_stems & gt_stems)
    
    if args.n > 0:
        matched = matched[:args.n]

    print(f"\nEnsemble 4-Panel Visualization")
    print(f"  Predictions : {pred_dir}")
    print(f"  Images      : {IMAGES_DIR}")
    print(f"  Ground Truth: {GT_DIR}")
    print(f"  Output      : {out_dir}")
    print(f"  Cases       : {len(matched)}")
    print()

    panel_paths = []
    all_dice = []
    for cid in matched:
        out = out_dir / f"{cid}_panel.png"
        try:
            dice_str, mean_d = make_panel(
                cid,
                IMAGES_DIR / f"{cid}_0000.nii.gz",
                GT_DIR     / f"{cid}.nii.gz",
                pred_dir   / f"{cid}.nii.gz",
                out
            )
            panel_paths.append(out)
            all_dice.append(mean_d)
        except Exception as e:
            print(f"  ✗ {cid}: {e}")

    if panel_paths:
        make_summary_grid(panel_paths[:12], out_dir)

    if all_dice:
        valid_dice = [d for d in all_dice if not np.isnan(d)]
        if valid_dice:
            print(f"\n{'='*60}")
            print(f"Overall Mean Dice: {np.mean(valid_dice):.4f}")
            print(f"Cases with valid Dice: {len(valid_dice)}/{len(matched)}")
            print(f"{'='*60}")

    print(f"\nDone. Open: {out_dir}/")

if __name__ == "__main__":
    main()
