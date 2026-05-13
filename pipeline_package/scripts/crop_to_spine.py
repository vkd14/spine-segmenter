#!/usr/bin/env python3
"""
Crop MRI volumes and segmentation masks to a tight 3D bounding box around L1-L5.

This removes all irrelevant anatomy (abdomen, soft tissue, pelvis etc.) so the
model only processes the vertebral column, dramatically reducing input size and
improving segmentation focus.

Pipeline:
  1. Load segmentation to find the 3D bounding box of all foreground labels
  2. Expand by a configurable margin (default 20mm in each direction)
  3. Crop both the MRI image and segmentation to this box
  4. Preserve NIfTI header / affine (update origin accordingly)

Usage:
    # Single case
    python scripts/crop_to_spine.py \
        --image  data/flair_inference_input/flair_100_..._0000.nii.gz \
        --seg    results/inference_raw_v2/flair_100_....nii.gz \
        --out-image  data/flair_cropped/flair_100_..._0000.nii.gz \
        --out-seg    results/inference_cropped/flair_100_....nii.gz

    # Batch mode (all cases)
    python scripts/crop_to_spine.py --batch \
        --image-dir  data/flair_inference_input/ \
        --seg-dir    results/inference_raw_v2/ \
        --out-image-dir  data/flair_cropped/ \
        --out-seg-dir    results/inference_cropped/ \
        --margin-mm  25 \
        --jobs 4
"""

import argparse
import numpy as np
import nibabel as nib
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed


def get_spine_bbox(seg_data: np.ndarray, voxel_sizes: tuple, margin_mm: float = 25.0):
    """
    Compute 3D bounding box of all foreground labels (>0) with a physical margin.
    Returns slices (sx, sy, sz) to index the volume.
    """
    fg = seg_data > 0
    if not fg.any():
        # No foreground — return full volume
        return tuple(slice(0, s) for s in seg_data.shape)

    coords = np.argwhere(fg)
    lo = coords.min(axis=0)
    hi = coords.max(axis=0)

    # Convert margin from mm to voxels per axis
    margin_vox = np.array([int(np.ceil(margin_mm / max(vs, 0.1))) for vs in voxel_sizes[:3]])

    lo = np.maximum(lo - margin_vox, 0)
    hi = np.minimum(hi + margin_vox + 1, np.array(seg_data.shape))

    slices = tuple(slice(int(l), int(h)) for l, h in zip(lo, hi))
    return slices, lo


def crop_image(img: nib.Nifti1Image, slices, lo: np.ndarray) -> nib.Nifti1Image:
    """Crop a NIfTI image to the given slices, updating the affine origin."""
    data = img.get_fdata(dtype=np.float32)[slices]
    aff = img.affine.copy()
    # Update origin: new_origin = old_origin + affine[:3,:3] @ lo
    new_origin = aff[:3, 3] + aff[:3, :3] @ lo.astype(float)
    aff[:3, 3] = new_origin
    return nib.Nifti1Image(data, aff, img.header)


def process_one(img_path: Path, seg_path: Path,
                out_img: Path, out_seg: Path,
                margin_mm: float = 25.0) -> str:
    try:
        seg_img = nib.load(str(seg_path))
        seg_data = seg_img.get_fdata(dtype=np.float32).astype(np.uint8)
        vox = seg_img.header.get_zooms()

        result = get_spine_bbox(seg_data, vox, margin_mm)
        if isinstance(result, tuple) and len(result) == 2:
            slices, lo = result
        else:
            slices = result
            lo = np.array([s.start for s in slices])

        crop_shape = tuple(s.stop - s.start for s in slices)

        # Crop seg
        seg_crop = seg_data[slices]
        seg_out = nib.Nifti1Image(seg_crop.astype(np.uint8), 
                                   crop_image(seg_img, slices, lo).affine,
                                   seg_img.header)
        out_seg.parent.mkdir(parents=True, exist_ok=True)
        nib.save(seg_out, str(out_seg))

        # Crop image if it exists
        if img_path and img_path.exists():
            img_nib = nib.load(str(img_path))
            img_crop = crop_image(img_nib, slices, lo)
            out_img.parent.mkdir(parents=True, exist_ok=True)
            nib.save(img_crop, str(out_img))

        n_fg = int((seg_crop > 0).sum())
        labels = sorted([l for l in range(1, 6) if (seg_crop == l).sum() > 0])
        return f"OK: {seg_path.name[:35]} → crop={crop_shape}, labels={labels}, fg={n_fg}"

    except Exception as e:
        return f"ERR: {seg_path.name[:35]}: {e}"


def main():
    ap = argparse.ArgumentParser()
    # Single mode
    ap.add_argument("--image",  type=Path, default=None)
    ap.add_argument("--seg",    type=Path, default=None)
    ap.add_argument("--out-image", type=Path, default=None)
    ap.add_argument("--out-seg",   type=Path, default=None)
    # Batch mode
    ap.add_argument("--batch",          action="store_true")
    ap.add_argument("--image-dir",      type=Path, default=Path("data/flair_inference_input"))
    ap.add_argument("--seg-dir",        type=Path, default=Path("results/inference_final_v2"))
    ap.add_argument("--out-image-dir",  type=Path, default=Path("data/flair_cropped"))
    ap.add_argument("--out-seg-dir",    type=Path, default=Path("results/seg_cropped"))
    ap.add_argument("--case-list",      type=Path, default=None,
                    help="Optional txt file listing filenames to process (from filter_pseudolabels.py)")
    ap.add_argument("--margin-mm",      type=float, default=25.0)
    ap.add_argument("--jobs",           type=int, default=4)
    args = ap.parse_args()

    if not args.batch:
        # Single case
        result = process_one(args.image, args.seg, args.out_image, args.out_seg, args.margin_mm)
        print(result)
        return

    # Batch mode
    seg_dir = args.seg_dir
    if args.case_list and args.case_list.exists():
        names = [n.strip() for n in open(args.case_list).readlines() if n.strip()]
        seg_files = [seg_dir / n for n in names if (seg_dir / n).exists()]
        print(f"Processing {len(seg_files)} cases from case list")
    else:
        seg_files = sorted(seg_dir.glob("*.nii.gz"))
        print(f"Processing all {len(seg_files)} cases in {seg_dir}")

    futures = {}
    done = errors = 0
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        for sf in seg_files:
            img_f   = args.image_dir    / sf.name if args.image_dir else None
            out_img = args.out_image_dir / sf.name if args.out_image_dir else None
            out_seg = args.out_seg_dir  / sf.name
            futures[ex.submit(process_one, img_f, sf, out_img, out_seg, args.margin_mm)] = sf

        for fut in as_completed(futures):
            msg = fut.result()
            if msg.startswith("OK"):
                done += 1
                if done % 20 == 0 or done <= 3:
                    print(f"  [{done}/{len(seg_files)}] {msg}")
            else:
                errors += 1
                print(f"  {msg}")

    print(f"\nDone: {done} cropped, {errors} errors")
    print(f"Images → {args.out_image_dir}")
    print(f"Segs   → {args.out_seg_dir}")


if __name__ == "__main__":
    main()
