"""
visualize_vertebra_panels.py
Generate 4-panel figures: Raw MRI | GT (Discs) | Ensemble Disc Pred | Vertebra Body Prediction

Since GT shows disc labels and the final output shows vertebral bodies,
Dice is computed between the disc prediction and GT (showing model accuracy),
while the vertebral body panel shows the derived clinical output.

Usage:
  python scripts/visualize_vertebra_panels.py [--n N]
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

BASE = Path(__file__).resolve().parent.parent
IMAGES_DIR = BASE / "data" / "nnUNet_raw" / "Dataset202_SpineL1L5_SSL" / "imagesTr"
GT_DIR     = BASE / "data" / "nnUNet_raw" / "Dataset202_SpineL1L5_SSL" / "labelsTr"
DISC_PRED_DIR  = BASE / "results" / "ensemble_flair_postprocessed"
BODY_PRED_DIR  = BASE / "results" / "ensemble_vertebra_bodies"
OUT_DIR        = BASE / "results" / "ensemble_vertebra_viz"

LABEL_NAMES = ["BG", "L1", "L2", "L3", "L4", "L5"]
COLORS      = ["none", "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4"]
SEG_CMAP    = ListedColormap(["none"] + COLORS[1:])
ALPHA       = 0.45


def load_volume(path):
    img = nib.load(str(path))
    data = img.get_fdata()
    ornt = nib_orient.io_orientation(img.affine)
    target_ornt = nib_orient.axcodes2ornt(('R', 'A', 'S'))
    transform = nib_orient.ornt_transform(ornt, target_ornt)
    return nib_orient.apply_orientation(data, transform)


def normalise(arr):
    lo, hi = np.percentile(arr, 1), np.percentile(arr, 99)
    return np.clip((arr - lo) / (hi - lo + 1e-8), 0, 1)


def overlay(ax, img2d, seg2d, cmap, title, title_color="white"):
    ax.imshow(img2d, cmap="gray", origin="lower", aspect="auto")
    masked = np.ma.masked_where(seg2d == 0, seg2d.astype(float))
    ax.imshow(masked, cmap=cmap, origin="lower", aspect="auto",
              alpha=ALPHA, vmin=0, vmax=5)
    ax.set_title(title, fontsize=10, fontweight="bold", color=title_color, pad=4)
    ax.axis("off")


def dice(gt2d, pred2d, cls):
    tp = ((gt2d == cls) & (pred2d == cls)).sum()
    denom = (gt2d == cls).sum() + (pred2d == cls).sum()
    return 2 * tp / denom if denom > 0 else float("nan")


def make_panel(case_id, img_path, gt_path, disc_pred_path, body_pred_path, out_path):
    img_vol  = load_volume(img_path)
    gt_vol   = load_volume(gt_path)
    disc_vol = load_volume(disc_pred_path)
    body_vol = load_volume(body_pred_path)

    # Match shapes
    s = tuple(min(a, b, c, d) for a, b, c, d in zip(
        img_vol.shape, gt_vol.shape, disc_vol.shape, body_vol.shape))
    img_vol  = img_vol[:s[0], :s[1], :s[2]]
    gt_vol   = gt_vol[:s[0], :s[1], :s[2]]
    disc_vol = disc_vol[:s[0], :s[1], :s[2]]
    body_vol = body_vol[:s[0], :s[1], :s[2]]

    # Best slice from GT
    gt_fg = (gt_vol > 0).reshape(gt_vol.shape[0], -1).sum(axis=1)
    idx = int(np.argmax(gt_fg)) if gt_fg.max() > 0 else gt_vol.shape[0] // 2

    img2d  = normalise(img_vol[idx])
    gt2d   = gt_vol[idx].astype(int)
    disc2d = disc_vol[idx].astype(int)
    body2d = body_vol[idx].astype(int)

    # Disc Dice (ensemble disc pred vs GT disc labels)
    dice_vals = [dice(gt2d, disc2d, c) for c in range(1, 6)]
    mean_dice = np.nanmean(dice_vals)

    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    fig.patch.set_facecolor("#111111")
    for ax in axes:
        ax.set_facecolor("#111111")

    # Panel 1: Raw MRI
    axes[0].imshow(img2d, cmap="gray", origin="lower", aspect="auto")
    axes[0].set_title("Raw MRI", fontsize=11, fontweight="bold", color="white", pad=4)
    axes[0].axis("off")

    # Panel 2: GT discs
    overlay(axes[1], img2d, gt2d, SEG_CMAP, "Ground Truth (Discs)")

    # Panel 3: Ensemble disc prediction
    overlay(axes[2], img2d, disc2d, SEG_CMAP, "Ensemble Disc Prediction")

    # Panel 4: Vertebral body prediction (derived from discs)
    overlay(axes[3], img2d, body2d, SEG_CMAP, "Vertebral Body (Derived)")

    # Legend
    patches = [mpatches.Patch(color=c, label=n)
               for c, n in zip(COLORS[1:], LABEL_NAMES[1:])]
    fig.legend(handles=patches, loc="lower center", ncol=5, fontsize=9,
               facecolor="#222222", edgecolor="none", labelcolor="white",
               bbox_to_anchor=(0.5, -0.02))

    # Title with disc Dice
    dice_str = [f"{LABEL_NAMES[c]}: {d:.2f}" for c, d in zip(range(1, 6), dice_vals)]
    fig.suptitle(
        f"{case_id}   |   Disc Dice — " + "  ".join(dice_str) + f"  |  Mean: {mean_dice:.2f}",
        fontsize=9, color="white", y=1.01)

    plt.tight_layout(pad=0.5)
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight", facecolor="#111111")
    plt.close(fig)
    print(f"  ✓ {out_path.name}  (disc Dice: {mean_dice:.3f})")
    return mean_dice


def make_summary_grid(panel_paths, out_dir):
    imgs = [plt.imread(str(p)) for p in panel_paths]
    cols = min(2, len(imgs))
    rows = (len(imgs) + cols - 1) // cols
    h, w = imgs[0].shape[:2]
    grid = np.ones((rows * h, cols * w, 3), dtype=np.float32)
    for i, im in enumerate(imgs):
        r, c = divmod(i, cols)
        if im.ndim == 2: im = np.stack([im]*3, axis=-1)
        if im.shape[2] == 4: im = im[..., :3]
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
    parser.add_argument("--n", type=int, default=0)
    parser.add_argument("--out-dir", type=str, default=str(OUT_DIR))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Find matching cases
    disc_stems = {p.name.replace(".nii.gz", "") for p in DISC_PRED_DIR.glob("*.nii.gz")}
    body_stems = {p.name.replace(".nii.gz", "") for p in BODY_PRED_DIR.glob("*.nii.gz")}
    img_stems  = {p.name.replace("_0000.nii.gz", "") for p in IMAGES_DIR.glob("*.nii.gz")}
    gt_stems   = {p.name.replace(".nii.gz", "") for p in GT_DIR.glob("*.nii.gz")}

    matched = sorted(disc_stems & body_stems & img_stems & gt_stems)
    if args.n > 0:
        matched = matched[:args.n]

    print(f"\n4-Panel Vertebra Body Visualization")
    print(f"  Disc Preds : {DISC_PRED_DIR}")
    print(f"  Body Preds : {BODY_PRED_DIR}")
    print(f"  Images     : {IMAGES_DIR}")
    print(f"  GT         : {GT_DIR}")
    print(f"  Output     : {out_dir}")
    print(f"  Cases      : {len(matched)}")
    print()

    panel_paths = []
    all_dice = []
    for cid in matched:
        out = out_dir / f"{cid}_panel.png"
        try:
            md = make_panel(
                cid,
                IMAGES_DIR / f"{cid}_0000.nii.gz",
                GT_DIR / f"{cid}.nii.gz",
                DISC_PRED_DIR / f"{cid}.nii.gz",
                BODY_PRED_DIR / f"{cid}.nii.gz",
                out
            )
            panel_paths.append(out)
            all_dice.append(md)
        except Exception as e:
            print(f"  ✗ {cid}: {e}")

    if panel_paths:
        make_summary_grid(panel_paths[:12], out_dir)

    if all_dice:
        valid = [d for d in all_dice if not np.isnan(d)]
        if valid:
            print(f"\n{'='*60}")
            print(f"Overall Disc Mean Dice: {np.mean(valid):.4f}")
            print(f"Cases visualized: {len(valid)}/{len(matched)}")
            print(f"{'='*60}")

    print(f"\nDone. Open: {out_dir}/")


if __name__ == "__main__":
    main()
