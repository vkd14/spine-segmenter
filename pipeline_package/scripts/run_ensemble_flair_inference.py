#!/usr/bin/env python3
"""
Ensemble Inference on T1-FLAIR Volumes
=======================================
Runs the 5-fold ensemble (Dataset202_SpineL1L5_SSL) on preprocessed
FLAIR volumes, applies post-processing, and generates visualisation panels.

Pipeline:
  1. nnUNetv2_predict  -f 0 1 2 3 4  (ensemble of all folds)
  2. Post-process each prediction (CC analysis, sagittal strip, vertical ordering)
  3. Generate per-case visualisation panels (Raw MRI + Segmentation overlay)

Usage:
    python scripts/run_ensemble_flair_inference.py [--n-vis 10]
"""

import os
import sys
import argparse
import subprocess
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

# ── paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent

# nnUNet env vars
os.environ["nnUNet_raw"] = str(ROOT / "data" / "nnUNet_raw")
os.environ["nnUNet_preprocessed"] = str(ROOT / "data" / "nnUNet_preprocessed")
os.environ["nnUNet_results"] = str(ROOT / "results" / "nnUNet_results")

INPUT_DIR   = ROOT / "data" / "flair_inference_input"          # preprocessed FLAIR _0000.nii.gz
RAW_PRED    = ROOT / "results" / "ensemble_flair_raw"          # raw nnUNet output
CLEAN_PRED  = ROOT / "results" / "ensemble_flair_postprocessed"# post-processed
VIS_DIR     = ROOT / "results" / "ensemble_flair_visualizations"

DATASET_ID  = 202
TRAINER     = "nnUNetTrainer_250epochs"
PLANS       = "nnUNetResEncUNetLPlans"
CONFIG      = "3d_fullres"
FOLDS       = "0 1 2 3 4"

# ── imports that might be heavy ────────────────────────────────────────────────
import nibabel as nib
import nibabel.orientations as nib_orient


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — nnU-Net ensemble prediction
# ══════════════════════════════════════════════════════════════════════════════

def run_ensemble_prediction(force: bool = False):
    """Run nnUNetv2_predict with all 5 folds (ensemble)."""
    RAW_PRED.mkdir(parents=True, exist_ok=True)

    # Check if already done
    existing = list(RAW_PRED.glob("*.nii.gz"))
    n_input  = len(list(INPUT_DIR.glob("*_0000.nii.gz")))
    if existing and len(existing) >= n_input and not force:
        print(f"✓ Ensemble predictions already exist ({len(existing)} files). Skipping.")
        return

    cmd = [
        "nnUNetv2_predict",
        "-i", str(INPUT_DIR),
        "-o", str(RAW_PRED),
        "-d", str(DATASET_ID),
        "-c", CONFIG,
        "-tr", TRAINER,
        "-p", PLANS,
        "-f", *FOLDS.split(),
        "-chk", "checkpoint_best.pth",
        "--disable_tta",
    ]

    print(f"\n{'='*70}")
    print(f"STEP 1: Ensemble prediction (5-fold) on {n_input} FLAIR volumes")
    print(f"  Dataset : {DATASET_ID}")
    print(f"  Trainer : {TRAINER}")
    print(f"  Plans   : {PLANS}")
    print(f"  Config  : {CONFIG}")
    print(f"  Folds   : {FOLDS}")
    print(f"  Input   : {INPUT_DIR}")
    print(f"  Output  : {RAW_PRED}")
    print(f"{'='*70}\n")
    print(" ".join(cmd), "\n")

    subprocess.run(cmd, check=True)
    n_out = len(list(RAW_PRED.glob("*.nii.gz")))
    print(f"\n✓ Ensemble prediction complete: {n_out} files in {RAW_PRED}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Post-processing
# ══════════════════════════════════════════════════════════════════════════════

# Inline the post-processing logic (from postprocess_segmentation.py)
from scipy.ndimage import label as cc_label

VERTEBRA_LABELS = [1, 2, 3, 4, 5]
LABEL_NAMES = {1: "L1", 2: "L2", 3: "L3", 4: "L4", 5: "L5"}


def clean_segmentation(seg, *, keep_largest_only=True, sagittal_strip_hw=0.20,
                        enforce_vertical_order=True, min_voxels=50, verbose=False):
    """Clean 3D segmentation: CC largest, sagittal strip, vertical ordering."""
    seg = seg.copy()
    shape = seg.shape

    fg = np.array(np.where(seg > 0)).T
    if len(fg) == 0:
        return seg

    spans = fg.max(axis=0) - fg.min(axis=0)
    lr_axis = int(np.argmin(spans))
    si_axis = int(np.argmax(spans))

    # Sagittal strip
    lr_min, lr_max = fg[:, lr_axis].min(), fg[:, lr_axis].max()
    lr_center = (lr_min + lr_max) / 2
    half_width = sagittal_strip_hw * shape[lr_axis]
    strip_lo = max(0, int(lr_center - half_width))
    strip_hi = min(shape[lr_axis] - 1, int(lr_center + half_width))

    strip_mask = np.zeros(shape, dtype=bool)
    slices = [slice(None)] * 3
    slices[lr_axis] = slice(strip_lo, strip_hi + 1)
    strip_mask[tuple(slices)] = True

    outside = (seg > 0) & ~strip_mask
    if verbose and outside.any():
        print(f"    Strip [{strip_lo}:{strip_hi}] axis {lr_axis}: removing {outside.sum()} voxels")
    seg[outside] = 0

    # Per-label largest CC
    centroids_si = {}
    for lab in VERTEBRA_LABELS:
        mask = seg == lab
        if not mask.any():
            continue
        labeled_arr, n_comp = cc_label(mask)
        if n_comp == 0:
            continue

        comp_sizes = [(cid, (labeled_arr == cid).sum()) for cid in range(1, n_comp + 1)]
        comp_sizes.sort(key=lambda x: x[1], reverse=True)

        best_id, best_size = comp_sizes[0]
        if best_size < min_voxels:
            seg[mask] = 0
            continue

        if keep_largest_only:
            for cid, sz in comp_sizes[1:]:
                seg[labeled_arr == cid] = 0
            mask = seg == lab

        coords = np.array(np.where(mask)).T
        centroids_si[lab] = coords[:, si_axis].mean()

    # Vertical ordering
    if enforce_vertical_order and len(centroids_si) >= 3:
        present = sorted(centroids_si.keys())
        si_vals = [centroids_si[l] for l in present]
        ascending = all(si_vals[i] < si_vals[i+1] for i in range(len(si_vals)-1))
        descending = all(si_vals[i] > si_vals[i+1] for i in range(len(si_vals)-1))

        if not ascending and not descending:
            direction = 1 if si_vals[-1] > si_vals[0] else -1
            prev = si_vals[0] * direction
            for lab, si in zip(present[1:], si_vals[1:]):
                if si * direction <= prev:
                    if verbose:
                        print(f"    Removing {LABEL_NAMES[lab]} (centroid out of order)")
                    seg[seg == lab] = 0
                else:
                    prev = si * direction

    return seg


def postprocess_one(pred_path, out_path, verbose=False):
    """Load, clean, save a single prediction NIfTI."""
    img = nib.load(str(pred_path))
    seg = img.get_fdata(dtype=np.float32).astype(np.int32)
    cleaned = clean_segmentation(seg, verbose=verbose)
    out_img = nib.Nifti1Image(cleaned.astype(np.uint8), img.affine, img.header)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(out_img, str(out_path))


def run_postprocessing(force: bool = False):
    """Post-process all raw predictions."""
    CLEAN_PRED.mkdir(parents=True, exist_ok=True)

    pred_files = sorted(RAW_PRED.glob("*.nii.gz"))
    existing   = set(f.name for f in CLEAN_PRED.glob("*.nii.gz"))
    todo       = [f for f in pred_files if force or f.name not in existing]

    print(f"\n{'='*70}")
    print(f"STEP 2: Post-processing {len(todo)}/{len(pred_files)} predictions")
    print(f"  Input  : {RAW_PRED}")
    print(f"  Output : {CLEAN_PRED}")
    print(f"{'='*70}\n")

    if not todo:
        print("✓ All already post-processed. Skipping.")
        return

    for i, pf in enumerate(todo):
        out = CLEAN_PRED / pf.name
        try:
            postprocess_one(pf, out, verbose=(i < 3))
            if (i + 1) % 20 == 0 or i < 3:
                print(f"  [{i+1}/{len(todo)}] {pf.name}")
        except Exception as e:
            print(f"  ERROR {pf.name}: {e}")

    final = len(list(CLEAN_PRED.glob("*.nii.gz")))
    print(f"\n✓ Post-processing complete: {final} clean segmentations in {CLEAN_PRED}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Visualisation
# ══════════════════════════════════════════════════════════════════════════════

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


def count_labels(seg):
    """Return dict of label→voxel_count for present labels."""
    return {LABEL_NAMES.get(l, f"?{l}"): int((seg == l).sum())
            for l in VERTEBRA_LABELS if (seg == l).any()}


def make_flair_panel(case_id, img_path, seg_path, out_path):
    """
    Generate a 3-panel visualisation:
      Panel 1: Raw MRI sagittal mid-slice
      Panel 2: Segmentation overlay (all labels)
      Panel 3: Per-vertebra contour overlay with labels
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.colors import ListedColormap

    COLORS_HEX = ["none", "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4"]
    COLORS_RGB  = [(230, 25, 75), (60, 180, 75), (67, 99, 216), (245, 130, 49), (145, 30, 180)]
    SEG_CMAP    = ListedColormap(["none"] + COLORS_HEX[1:])
    ALPHA       = 0.45

    img_vol = load_volume_ras(img_path)
    seg_vol = load_volume_ras(seg_path)

    # Ensure matching shapes by trimming if needed
    min_shape = tuple(min(a, b) for a, b in zip(img_vol.shape, seg_vol.shape))
    img_vol = img_vol[:min_shape[0], :min_shape[1], :min_shape[2]]
    seg_vol = seg_vol[:min_shape[0], :min_shape[1], :min_shape[2]]

    # Find best slice: axis 0 (sagittal in RAS)
    fg_per_slice = (seg_vol > 0).reshape(seg_vol.shape[0], -1).sum(axis=1)
    if fg_per_slice.max() == 0:
        # No foreground — use middle slice
        idx = seg_vol.shape[0] // 2
    else:
        idx = int(np.argmax(fg_per_slice))

    img2d = normalise(img_vol[idx])
    seg2d = seg_vol[idx].astype(int)

    # Present vertebrae
    present = count_labels(seg_vol)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.patch.set_facecolor("#111111")
    for ax in axes:
        ax.set_facecolor("#111111")

    # P1: Raw MRI
    axes[0].imshow(img2d, cmap="gray", origin="lower", aspect="auto")
    axes[0].set_title("Raw FLAIR MRI", fontsize=11, fontweight="bold", color="white", pad=4)
    axes[0].axis("off")

    # P2: Segmentation overlay
    axes[1].imshow(img2d, cmap="gray", origin="lower", aspect="auto")
    masked = np.ma.masked_where(seg2d == 0, seg2d.astype(float))
    axes[1].imshow(masked, cmap=SEG_CMAP, origin="lower", aspect="auto",
                   alpha=ALPHA, vmin=0, vmax=5)
    axes[1].set_title("Ensemble Segmentation", fontsize=11, fontweight="bold",
                      color="white", pad=4)
    axes[1].axis("off")

    # P3: Contour overlay with labels
    import cv2
    norm_u8 = (img2d * 255).astype(np.uint8)
    overlay_bgr = cv2.cvtColor(norm_u8, cv2.COLOR_GRAY2BGR)

    for lab_val, color in zip(VERTEBRA_LABELS, COLORS_RGB):
        vert_mask = (seg2d == lab_val).astype(np.uint8)
        if vert_mask.sum() == 0:
            continue
        # Semi-transparent fill
        color_mask = np.zeros_like(overlay_bgr)
        color_mask[vert_mask > 0] = color[::-1]  # RGB→BGR
        overlay_bgr = cv2.addWeighted(overlay_bgr, 1.0, color_mask, 0.35, 0)
        # Contours
        contours, _ = cv2.findContours(vert_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay_bgr, contours, -1, color[::-1], 2)
        # Label text
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
    axes[2].set_title("Vertebra Labels", fontsize=11, fontweight="bold", color="white", pad=4)
    axes[2].axis("off")

    # Legend
    patches = [mpatches.Patch(color=c, label=n) for c, n in zip(COLORS_HEX[1:],
               ["L1", "L2", "L3", "L4", "L5"])]
    fig.legend(handles=patches, loc="lower center", ncol=5, fontsize=10,
               facecolor="#222222", edgecolor="none", labelcolor="white",
               bbox_to_anchor=(0.5, -0.02))

    # Title with detected vertebrae
    det_str = ", ".join(f"{k}: {v:,} vox" for k, v in present.items()) if present else "No vertebrae detected"
    fig.suptitle(f"{case_id}   |   Detected: {det_str}",
                 fontsize=10, color="white", y=1.01)

    plt.tight_layout(pad=0.5)
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight", facecolor="#111111")
    plt.close(fig)
    return present


def make_summary_grid(panel_paths, out_path):
    """Combine individual panels into a summary grid image."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

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
        grid[r*h:(r+1)*h, c*w:(c+1)*w] = im[:h, :w]

    fig, ax = plt.subplots(figsize=(cols * 10, rows * 4))
    fig.patch.set_facecolor("#111111")
    ax.imshow(grid)
    ax.axis("off")
    plt.tight_layout(pad=0)
    fig.savefig(str(out_path), dpi=120, bbox_inches="tight", facecolor="#111111")
    plt.close(fig)
    print(f"✓ Summary grid → {out_path}")


def run_visualisations(n_vis: int = 10):
    """Generate visualisations for post-processed ensemble predictions."""
    VIS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"STEP 3: Generating visualisations ({n_vis} cases)")
    print(f"  Input images : {INPUT_DIR}")
    print(f"  Segmentations: {CLEAN_PRED}")
    print(f"  Output       : {VIS_DIR}")
    print(f"{'='*70}\n")

    # Match seg files to input images
    seg_files = sorted(CLEAN_PRED.glob("*.nii.gz"))
    panel_paths = []
    n_done = 0

    for sf in seg_files:
        if n_done >= n_vis:
            break
        case_id = sf.name.replace(".nii.gz", "")
        img_file = INPUT_DIR / f"{case_id}_0000.nii.gz"
        if not img_file.exists():
            continue

        out_png = VIS_DIR / f"{case_id}_ensemble_panel.png"
        try:
            present = make_flair_panel(case_id, img_file, sf, out_png)
            n_vert = len(present)
            print(f"  ✓ {case_id} — {n_vert} vertebrae detected")
            panel_paths.append(out_png)
            n_done += 1
        except Exception as e:
            print(f"  ✗ {case_id}: {e}")

    if panel_paths:
        grid_path = VIS_DIR / "ensemble_summary_grid.png"
        make_summary_grid(panel_paths, grid_path)

    print(f"\n✓ Visualisations complete: {n_done} panels in {VIS_DIR}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Ensemble FLAIR Inference Pipeline")
    parser.add_argument("--n-vis", type=int, default=10,
                        help="Number of cases to visualise (default: 10)")
    parser.add_argument("--force", action="store_true",
                        help="Force re-run even if outputs exist")
    parser.add_argument("--skip-predict", action="store_true",
                        help="Skip nnUNet prediction (use existing raw predictions)")
    parser.add_argument("--skip-postprocess", action="store_true",
                        help="Skip post-processing (use existing cleaned predictions)")
    parser.add_argument("--only-vis", action="store_true",
                        help="Only run visualisation step")
    args = parser.parse_args()

    print(f"\n{'#'*70}")
    print(f"#  Ensemble FLAIR Inference Pipeline")
    print(f"#  Dataset: {DATASET_ID} | Trainer: {TRAINER}")
    print(f"#  Folds: {FOLDS}")
    print(f"{'#'*70}\n")

    if not args.only_vis:
        if not args.skip_predict:
            run_ensemble_prediction(force=args.force)

        if not args.skip_postprocess:
            run_postprocessing(force=args.force)

    run_visualisations(n_vis=args.n_vis)

    print(f"\n{'='*70}")
    print(f"Pipeline complete!")
    print(f"  Raw predictions   : {RAW_PRED}")
    print(f"  Post-processed    : {CLEAN_PRED}")
    print(f"  Visualisations    : {VIS_DIR}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
