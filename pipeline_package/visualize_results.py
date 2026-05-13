"""
visualize_results.py
Generate presentation-quality figures comparing:
  - Raw MRI (sagittal mid-slice)
  - Ground-truth segmentation overlay
  - Model prediction overlay
  - Correct / Error map

Notes on data dimensions:
  IMG/GT:  (S, H, W) = (17, 512, 512)   -- S = sagittal slice index
  PRED:    (H, W, S) = (512, 512, 17)   -- nnUNet stores rotated; transpose to match

Outputs:
  results/visualizations/<case>_panel.png
  results/visualizations/summary_grid.png

Usage:
  python visualize_results.py [--n N]
"""

import argparse, re
import numpy as np
import nibabel as nib
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from pathlib import Path

BASE       = Path(__file__).parent
IMAGES_DIR = BASE / "data/nnUNet_raw/Dataset202_SpineL1L5_SSL/imagesTr"
GT_DIR     = BASE / "data/nnUNet_raw/Dataset202_SpineL1L5_SSL/labelsTr"
PRED_DIR   = BASE / "results/inference_final"
OUT_DIR    = BASE / "results/visualizations"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LABEL_NAMES = ["BG", "L1", "L2", "L3", "L4", "L5"]
COLORS      = ["none", "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4"]
SEG_CMAP    = ListedColormap(["none"] + COLORS[1:])   # 6 entries, index 0=transparent
ALPHA       = 0.45

def load_volume(path):
    img = nib.load(str(path))
    data = img.get_fdata()
    # Reorient to RAS+ to ensure consistent axis alignment
    ornt = nib.orientations.io_orientation(img.affine)
    target_ornt = nib.orientations.axcodes2ornt(('R', 'A', 'S'))
    transform = nib.orientations.ornt_transform(ornt, target_ornt)
    return nib.orientations.apply_orientation(data, transform)

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

    assert img_vol.shape == gt_vol.shape == pred_vol.shape, \
        f"Shape mismatch: img={img_vol.shape} gt={gt_vol.shape} pred={pred_vol.shape}"

    idx     = best_slice_idx(gt_vol)
    img2d   = normalise(img_vol[idx])
    gt2d    = gt_vol[idx].astype(int)
    pred2d  = pred_vol[idx].astype(int)

    # Diff: 1=correct foreground, 2=wrong class
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
    overlay(axes[1], img2d, gt2d,   SEG_CMAP, "Ground Truth")

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
               for c, n in zip(COLORS[1:], LABEL_NAMES[1:])]
    fig.legend(handles=patches, loc="lower center", ncol=5, fontsize=9,
               facecolor="#222222", edgecolor="none", labelcolor="white",
               bbox_to_anchor=(0.5, -0.02))

    # Per-class Dice in title
    dice_str = [f"{LABEL_NAMES[c]}: {dice(gt2d, pred2d, c):.2f}" for c in range(1, 6)]
    fig.suptitle(f"{case_id}   |   Dice — " + "  ".join(dice_str),
                 fontsize=9, color="white", y=1.01)

    plt.tight_layout(pad=0.5)
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight",
                facecolor="#111111")
    plt.close(fig)
    print(f"  ✓ {out_path.name}")
    return dice_str

def make_summary_grid(panel_paths):
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
    out = OUT_DIR / "summary_grid.png"
    fig, ax = plt.subplots(figsize=(cols * 10, rows * 4))
    fig.patch.set_facecolor("#111111")
    ax.imshow(grid); ax.axis("off")
    plt.tight_layout(pad=0)
    fig.savefig(str(out), dpi=120, bbox_inches="tight", facecolor="#111111")
    plt.close(fig)
    print(f"\n✓ Summary grid → {out}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=6)
    args = parser.parse_args()

    # Find matching cases
    pred_stems = {p.name.replace(".nii.gz","") for p in PRED_DIR.glob("*.nii.gz")}
    img_stems  = {p.name.replace("_0000.nii.gz","") for p in IMAGES_DIR.glob("*.nii.gz")}
    matched = sorted(pred_stems & img_stems)[:args.n]

    print(f"Visualising {len(matched)} cases → {OUT_DIR}\n")
    panel_paths = []
    for cid in matched:
        out = OUT_DIR / f"{cid}_panel.png"
        try:
            make_panel(cid,
                       IMAGES_DIR / f"{cid}_0000.nii.gz",
                       GT_DIR     / f"{cid}.nii.gz",
                       PRED_DIR   / f"{cid}.nii.gz",
                       out)
            panel_paths.append(out)
        except Exception as e:
            print(f"  ✗ {cid}: {e}")

    if panel_paths:
        make_summary_grid(panel_paths)
    print(f"\nDone. Open: results/visualizations/")

if __name__ == "__main__":
    main()
