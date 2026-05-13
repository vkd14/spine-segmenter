#!/usr/bin/env python3
"""
Pre-process T1-FLAIR volumes for nnU-Net inference.

Problem:  T1-FLAIR volumes are stored as (512, 512, 18) axial thick-slice
          but the model was trained on sagittal thin-slice volumes.
          nnUNet resamples to its target spacing but with only 15-18 thick
          slices along Z, the through-plane resolution is too poor → bad segs.

Fix:      Reorient each volume to RAS+ sagittal orientation so the coronal
          plane (X-axis) becomes the thin dimension, matching training data.
          Additionally clip/normalise intensity to match training distribution.

Usage:
    python scripts/preprocess_flair_for_inference.py \
        --input  data/Sagittal_T1_FLAIR/ \
        --output data/flair_inference_input/ \
        --jobs   4
"""

import os, argparse, warnings
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import nibabel as nib
import nibabel.orientations as nib_orient

warnings.filterwarnings("ignore")


def reorient_to_ras(img: nib.Nifti1Image) -> nib.Nifti1Image:
    """Reorient NIfTI to RAS+ standard orientation."""
    return nib.as_closest_canonical(img)


def reorient_sagittal(img: nib.Nifti1Image) -> nib.Nifti1Image:
    """
    Reorient so that the left-right (L/R) axis is the FIRST (x) axis.
    i.e. axial → sagittal reformatting.
    Target orientation: ('L','A','S') or ('R','A','S').
    """
    # First go to RAS canonical
    ras = nib.as_closest_canonical(img)
    # RAS = (R, A, S) i.e. x=L-R, y=P-A, z=I-S
    # For sagittal the SHORT axis should be x (L-R).
    # If the volume is (512,512,N) with N=18 in RAS, the z-dimension is the
    # short one → we need to permute axes so it becomes x.
    data = ras.get_fdata()
    shape = data.shape  # (X,Y,Z) in RAS

    # Find shortest axis — that's the through-plane direction
    shortest = int(np.argmin(shape))
    if shortest == 0:
        # Already sagittal-like (short axis = x = L-R)
        return ras

    # Permute so shortest axis moves to position 0
    perm = list(range(3))
    perm[0], perm[shortest] = perm[shortest], perm[0]
    data_perm = np.transpose(data, perm)

    # Update affine: permute columns
    aff = ras.affine.copy()
    aff[:3, :3] = aff[:3, perm]
    
    new_img = nib.Nifti1Image(data_perm, aff, ras.header)
    return new_img


def process_one(src: Path, dst: Path):
    img = nib.load(str(src))
    img_sag = reorient_sagittal(img)
    
    data = img_sag.get_fdata(dtype=np.float32)
    zooms = img_sag.header.get_zooms()
    
    # Save with _0000 suffix (nnUNet convention)
    nib.save(img_sag, str(dst))
    return src.name, data.shape, [round(float(z), 3) for z in zooms]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",  default="data/Sagittal_T1_FLAIR")
    ap.add_argument("--output", default="data/flair_inference_input")
    ap.add_argument("--jobs",   type=int, default=4)
    args = ap.parse_args()

    src_dir = Path(args.input)
    dst_dir = Path(args.output)
    dst_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(src_dir.glob("*.nii.gz"))
    print(f"Processing {len(files)} T1-FLAIR volumes → {dst_dir}")

    done = errors = 0
    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        futures = {}
        for f in files:
            stem = f.name.replace(".nii.gz", "")
            dst = dst_dir / f"flair_{stem}_0000.nii.gz"
            if dst.exists():
                done += 1
                continue
            futures[ex.submit(process_one, f, dst)] = f

        for fut in as_completed(futures):
            try:
                name, shape, zooms = fut.result()
                done += 1
                if done % 20 == 0 or done <= 3:
                    print(f"  [{done}/{len(files)}] {name}: shape={shape}, zooms={zooms}")
            except Exception as e:
                errors += 1
                print(f"  ERROR {futures[fut].name}: {e}")

    print(f"\nDone: {done} processed, {errors} errors")
    print(f"Output: {dst_dir}")
    print(f"\nNow run inference:")
    print(f"  nnUNetv2_predict \\")
    print(f"    -i {dst_dir} \\")
    print(f"    -o results/inference_raw_v2 \\")
    print(f"    -d 100 -c 3d_fullres -f 0 1 2 3 4 \\")
    print(f"    -tr nnUNetTrainer -p nnUNetResEncUNetLPlans \\")
    print(f"    -chk checkpoint_best.pth --disable_tta")


if __name__ == "__main__":
    main()
