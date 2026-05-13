"""
Post-processing for nnU-Net L1-L5 spine segmentation outputs.

Problems addressed:
  1. Spurious "mirror" segmentation blobs on the contralateral side of the spine
     (network hallucinates vertebra tissue at a reflection of the true vertebra).
  2. Disconnected small fragments far from the main vertebra body.

Strategy (per-label):
  a. Connected-component analysis → keep only the LARGEST component per label.
  b. Anatomy constraint: all vertebrae must lie within the central sagittal strip
     (left-right axis). Any labeled voxel outside [cx-hw, cx+hw] is zeroed.
  c. Optional: enforce vertical ordering — L1 centroid must be ABOVE L2, etc.
     If violated, relabel or discard the offending component.

Usage:
    # Post-process a single prediction file
    python scripts/postprocess_segmentation.py predicted.nii.gz -o cleaned.nii.gz

    # Post-process a whole directory
    python scripts/postprocess_segmentation.py ./predictions/ -o ./predictions_clean/

    # As part of run_inference.py (import and call clean_segmentation())
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path

try:
    import nibabel as nib
except ImportError:
    raise ImportError("pip install nibabel")

try:
    from scipy.ndimage import label as cc_label, center_of_mass
except ImportError:
    raise ImportError("pip install scipy")


VERTEBRA_LABELS = [1, 2, 3, 4, 5]   # L1 → L5
LABEL_NAMES = {1: "L1", 2: "L2", 3: "L3", 4: "L4", 5: "L5"}


# ─── Core cleaning function ───────────────────────────────────────────────────

def clean_segmentation(
    seg: np.ndarray,
    affine: np.ndarray = None,
    *,
    keep_largest_only: bool = True,
    sagittal_strip_hw: float = 0.20,   # half-width as fraction of image width on LR axis
    enforce_vertical_order: bool = True,
    min_voxels: int = 50,
    verbose: bool = False,
) -> np.ndarray:
    """
    Clean a 3D segmentation array (labels 0-5, 0=background).

    Parameters
    ----------
    seg : ndarray (H, W, D) int
    affine : optional 4x4 affine (unused currently, reserved for mm-space constraints)
    keep_largest_only : remove all but largest connected component per label
    sagittal_strip_hw : fraction of the LR image width to keep around the spine midline
    enforce_vertical_order : ensure L1 is superior to L5, swap/remove if not
    min_voxels : discard connected components smaller than this
    verbose : print per-label stats

    Returns
    -------
    cleaned ndarray, same shape and dtype as input
    """
    seg = seg.copy()
    shape = seg.shape  # (AP, LR, SI) or any orientation — we work on raw voxel coords

    # ── Step 1: Find the LR axis (where spine sits centrally) ────────────────
    # Heuristic: spine labels are distributed narrowly on one axis (LR) and
    # spread on the other two. We find the axis with the smallest span of labelled voxels.
    fg = np.array(np.where(seg > 0)).T   # (N, 3)
    if len(fg) == 0:
        return seg

    spans = fg.max(axis=0) - fg.min(axis=0)
    lr_axis = int(np.argmin(spans))      # smallest span = left-right axis
    si_axis = int(np.argmax(spans))      # largest span  = superior-inferior axis
    # ap_axis = the remaining one

    # ── Step 2: Sagittal strip constraint (remove contralateral blobs) ────────
    lr_min, lr_max = fg[:, lr_axis].min(), fg[:, lr_axis].max()
    lr_center = (lr_min + lr_max) / 2
    half_width = sagittal_strip_hw * shape[lr_axis]

    strip_lo = max(0, int(lr_center - half_width))
    strip_hi = min(shape[lr_axis] - 1, int(lr_center + half_width))

    # Build mask: only keep voxels within the central strip
    strip_mask = np.zeros(shape, dtype=bool)
    slices = [slice(None)] * 3
    slices[lr_axis] = slice(strip_lo, strip_hi + 1)
    strip_mask[tuple(slices)] = True

    outside_strip = (seg > 0) & ~strip_mask
    if verbose and outside_strip.any():
        print(f"  Strip [{strip_lo}:{strip_hi}] on axis {lr_axis}: "
              f"removing {outside_strip.sum()} voxels outside spine corridor")
    seg[outside_strip] = 0

    # ── Step 3: Per-label largest connected component ──────────────────────────
    centroids_si = {}   # label → SI coordinate of centroid

    for lab in VERTEBRA_LABELS:
        mask = seg == lab
        if not mask.any():
            continue

        labeled_arr, n_components = cc_label(mask)
        if n_components == 0:
            continue

        # Count component sizes
        comp_sizes = [(comp_id, (labeled_arr == comp_id).sum())
                      for comp_id in range(1, n_components + 1)]
        comp_sizes.sort(key=lambda x: x[1], reverse=True)

        if verbose and n_components > 1:
            print(f"  {LABEL_NAMES[lab]}: {n_components} components → "
                  f"sizes {[s for _, s in comp_sizes]}")

        # zero everything except the largest (if above min_voxels)
        best_id, best_size = comp_sizes[0]
        if best_size < min_voxels:
            seg[mask] = 0
            if verbose:
                print(f"  {LABEL_NAMES[lab]}: largest component only {best_size} voxels — discarding")
            continue

        # Keep only the largest component
        if keep_largest_only:
            for comp_id, sz in comp_sizes[1:]:
                seg[labeled_arr == comp_id] = 0
            # Recompute mask after removal for centroid
            mask = seg == lab

        # Record centroid on the SI axis
        coords = np.array(np.where(mask)).T
        centroids_si[lab] = coords[:, si_axis].mean()

    # ── Step 4: Vertical ordering check ──────────────────────────────────────
    # In a standard orientation, L1 is SUPERIOR (smaller SI index if SI↑ or
    # larger if SI↓). We use relative ordering: L1 centroid < L2 < ... < L5
    # on the SI axis (regardless of direction) — if the majority ordering
    # suggests ascending, L1 should be at the low end.
    if enforce_vertical_order and len(centroids_si) >= 3:
        present = sorted(centroids_si.keys())
        si_vals = [centroids_si[l] for l in present]
        ascending = all(si_vals[i] < si_vals[i+1] for i in range(len(si_vals)-1))
        descending = all(si_vals[i] > si_vals[i+1] for i in range(len(si_vals)-1))

        if not ascending and not descending:
            # Find the offending label(s)
            if verbose:
                print(f"  Vertebra order violation: {dict(zip(present, si_vals))}")
            # Remove any label whose centroid breaks monotonic ordering
            direction = 1 if si_vals[-1] > si_vals[0] else -1
            prev = si_vals[0] * direction
            for lab, si in zip(present[1:], si_vals[1:]):
                if si * direction <= prev:
                    if verbose:
                        print(f"  Removing {LABEL_NAMES[lab]} (centroid out of order)")
                    seg[seg == lab] = 0
                else:
                    prev = si * direction

    return seg


# ─── File I/O helpers ──────────────────────────────────────────────────────────

def process_file(in_path: Path, out_path: Path, verbose: bool = False):
    img = nib.load(str(in_path))
    seg = img.get_fdata(dtype=np.float32).astype(np.int32)
    header = img.header

    before_counts = {l: (seg == l).sum() for l in VERTEBRA_LABELS}
    cleaned = clean_segmentation(seg, affine=img.affine, verbose=verbose)
    after_counts = {l: (cleaned == l).sum() for l in VERTEBRA_LABELS}

    if verbose:
        print(f"\n  Voxel counts (before → after):")
        for l in VERTEBRA_LABELS:
            removed = before_counts[l] - after_counts[l]
            pct = (removed / before_counts[l] * 100) if before_counts[l] > 0 else 0
            print(f"    {LABEL_NAMES[l]}: {before_counts[l]:>7,} → {after_counts[l]:>7,}  "
                  f"({removed:+,} / {pct:.1f}% removed)")

    out_img = nib.Nifti1Image(cleaned.astype(np.uint8), img.affine, header)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(out_img, str(out_path))
    print(f"  Saved: {out_path}")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Post-process nnU-Net spine segmentations: remove mirror blobs, "
                    "keep only largest component per vertebra, enforce anatomical ordering."
    )
    parser.add_argument("input", help="Input NIfTI file or directory of predictions")
    parser.add_argument("-o", "--output", default=None,
                        help="Output file or directory (default: <input>_clean)")
    parser.add_argument("--strip-hw", type=float, default=0.20,
                        help="Sagittal strip half-width as fraction of image width (default: 0.20). "
                             "Decrease if false positives are close to the spine.")
    parser.add_argument("--min-voxels", type=int, default=50,
                        help="Minimum component size to keep (default: 50)")
    parser.add_argument("--no-order-check", action="store_true",
                        help="Disable vertical ordering enforcement")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    in_path = Path(args.input)

    if in_path.is_file():
        out_path = Path(args.output) if args.output else in_path.with_name(
            in_path.name.replace(".nii.gz", "_clean.nii.gz").replace(".nii", "_clean.nii"))
        print(f"\n{'─'*60}")
        print(f"  Input:  {in_path}")
        print(f"  Output: {out_path}")
        process_file(in_path, out_path,
                     verbose=args.verbose)

    elif in_path.is_dir():
        out_dir = Path(args.output) if args.output else in_path.parent / (in_path.name + "_clean")
        files = sorted(in_path.glob("*.nii.gz")) + sorted(in_path.glob("*.nii"))
        files = [f for f in files if "_clean" not in f.name]
        print(f"\nProcessing {len(files)} files → {out_dir}")
        for f in files:
            out_path = out_dir / f.name
            print(f"\n{'─'*60}")
            print(f"  {f.name}")
            process_file(f, out_path, verbose=args.verbose)

    else:
        print(f"ERROR: {in_path} not found.")
        sys.exit(1)

    print(f"\n{'─'*60}")
    print("Post-processing complete.")


if __name__ == "__main__":
    main()
