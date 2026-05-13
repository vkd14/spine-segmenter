#!/usr/bin/env python3
"""
cascade_inference.py — 2-stage cascaded inference pipeline

Stage 1: Binary spine model (Dataset200) → bounding box of spine
Stage 2: L1-L5 model (Dataset201) → vertebra segmentation on cropped ROI → restored to full space

Usage:
    python scripts/cascade_inference.py \
        --input  data/Sagittal_T1_FLAIR/ \
        --output results/cascade_final/ \
        --stage1-dataset 200 \
        --stage2-dataset 201

Or run inference with existing pseudo-label stage1 masks:
    python scripts/cascade_inference.py \
        --input  data/Sagittal_T1_FLAIR/ \
        --output results/cascade_final/ \
        --stage1-masks  results/stage1_binary/  \  # skip stage1 if already done
        --stage2-dataset 201
"""

import os, sys, shutil, tempfile, argparse
from pathlib import Path
import numpy as np
import nibabel as nib
import SimpleITK as sitk

ROOT = Path(__file__).parent.parent


def get_spine_bbox(seg_arr, pad_mm, spacing_xyz):
    """
    Return (xmin,xmax, ymin,ymax, zmin,zmax) bbox of nonzero voxels
    with pad_mm padding applied.
    """
    nz = np.argwhere(seg_arr > 0)
    if len(nz) == 0:
        return None
    lo = nz.min(axis=0)   # shape=(3,) → x,y,z
    hi = nz.max(axis=0)

    # Guard against zero spacing (oblique affines can have zero diagonal)
    safe_spacing = np.where(spacing_xyz > 1e-6, spacing_xyz, 1.0)
    pad_vox = [int(np.ceil(pad_mm / safe_spacing[i])) for i in range(3)]

    lo = np.maximum(lo - pad_vox, 0)
    hi = np.minimum(hi + pad_vox, np.array(seg_arr.shape) - 1)
    return lo, hi


def crop_nib(img_nib, lo, hi):
    """Crop a nibabel image to the given bounding box (voxel coords)."""
    arr = np.array(img_nib.dataobj)
    cropped = arr[lo[0]:hi[0]+1, lo[1]:hi[1]+1, lo[2]:hi[2]+1]

    # Update affine: shift origin
    new_affine = img_nib.affine.copy()
    new_affine[:3, 3] = img_nib.affine[:3, :3] @ lo + img_nib.affine[:3, 3]
    return nib.Nifti1Image(cropped.astype(arr.dtype), new_affine, img_nib.header)


def restore_to_full(cropped_seg, full_img, lo, hi):
    """Paste cropped segmentation back into full-volume zeros."""
    full_arr = np.zeros(np.array(full_img.dataobj).shape, dtype=np.int16)
    seg_arr  = np.array(cropped_seg.dataobj, dtype=np.int16)
    full_arr[lo[0]:hi[0]+1, lo[1]:hi[1]+1, lo[2]:hi[2]+1] = seg_arr
    return nib.Nifti1Image(full_arr, full_img.affine, full_img.header)


def run_nnunet_inference(input_dir, output_dir, dataset_id, fold="all",
                          trainer="nnUNetTrainer",
                          plans="nnUNetResEncUNetLPlans",
                          config="3d_fullres",
                          step_size=0.5, disable_tta=False):
    """Call nnUNetv2_predict."""
    import subprocess

    results_root = str(ROOT / "results" / "nnUNet_results")
    env = {**os.environ,
           "nnUNet_raw":         str(ROOT / "data" / "nnUNet_raw"),
           "nnUNet_preprocessed":str(ROOT / "data" / "nnUNet_preprocessed"),
           "nnUNet_results":     results_root,
           "nnUNet_compile":     "0"}

    # Build fold args
    if isinstance(fold, str) and fold == "all":
        fold_args = ["0", "1", "2", "3", "4"]
    elif isinstance(fold, str):
        fold_args = fold.split(",")
    else:
        fold_args = [str(fold)]

    cmd = [
        "nnUNetv2_predict",
        "-i", str(input_dir),
        "-o", str(output_dir),
        "-d", str(dataset_id),
        "-tr", trainer,
        "-p", plans,
        "-c", config,
        "-f", *fold_args,
        "-chk", "checkpoint_best.pth",
        "-step_size", str(step_size),
    ]
    if disable_tta:
        cmd.append("--disable_tta")

    print(f"  Running: {' '.join(cmd)}")
    subprocess.run(cmd, env=env, check=True)


def main():
    parser = argparse.ArgumentParser(description="2-stage cascaded spine inference")
    parser.add_argument("--input",  required=True, help="Dir of raw NIfTI images (not formatted)")
    parser.add_argument("--output", required=True, help="Output dir for final segmentations")
    parser.add_argument("--stage1-dataset", type=int, default=200)
    parser.add_argument("--stage2-dataset", type=int, default=201)
    parser.add_argument("--stage1-trainer", default="nnUNetTrainer_100epochs", help="Trainer for Stage1")
    parser.add_argument("--stage2-trainer", default="nnUNetTrainer_250epochs", help="Trainer for Stage2")
    parser.add_argument("--stage1-masks",   help="Skip Stage1 — use existing binary masks from here")
    parser.add_argument("--pad-mm",  type=float, default=30.0, help="Padding (mm) around spine bbox")
    parser.add_argument("--fold",    default="all", help="Stage2 folds to use")
    parser.add_argument("--disable-tta", action="store_true")
    args = parser.parse_args()

    input_dir  = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect input images
    images = sorted(input_dir.glob("*.nii.gz"))
    if not images:
        images = sorted(input_dir.glob("*.nii"))
    print(f"Found {len(images)} input images")

    # Use persistent stage1 dir to avoid re-running on retry
    stage1_persist = output_dir / "_stage1_binary"

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        s1_input  = tmp / "s1_input"
        s1_output = stage1_persist  # persistent
        s2_input  = tmp / "s2_input"
        s2_output = tmp / "s2_output"
        s1_input.mkdir()
        s2_input.mkdir()

        # ── Format input for nnUNet (_0000 suffix) ────────────────────────────
        case_map = {}  # nnunet_name → original Path
        for img_path in images:
            stem = img_path.stem.replace(".nii", "")
            case_name = stem if "_0000" in stem else f"{stem}"
            nnunet_name = case_name if stem.endswith("_0000") else f"{stem}_0000"
            dst = s1_input / f"{nnunet_name}.nii.gz"
            if not dst.exists():
                shutil.copy2(img_path, dst)
            case_map[case_name] = img_path

        # ── Stage 1: Binary spine localization ────────────────────────────────
        if args.stage1_masks:
            s1_output = Path(args.stage1_masks)
            print(f"[Stage1] Using existing masks from {s1_output}")
        else:
            print(f"\n[Stage1] Binary spine inference...")
            s1_output.mkdir()
            run_nnunet_inference(
                s1_input, s1_output,
                dataset_id=args.stage1_dataset,
                fold="0",  # single fold for speed
                trainer=args.stage1_trainer,
                disable_tta=True,  # binary is easy, TTA not needed
            )

        # ── Crop + format for Stage 2 ─────────────────────────────────────────
        print(f"\n[Crop] Cropping to spine ROI (pad={args.pad_mm}mm)...")
        bbox_map = {}  # case_name → (original_img_path, lo, hi)

        for img_path in images:
            stem = img_path.stem.replace(".nii", "")
            case_name = stem if not stem.endswith("_0000") else stem[:-5]

            # Find Stage1 output
            s1_seg_path = s1_output / f"{case_name}.nii.gz"
            if not s1_seg_path.exists():
                s1_seg_path = s1_output / f"{case_name}_0000.nii.gz"
            if not s1_seg_path.exists():
                print(f"  SKIP {case_name} — no Stage1 output")
                continue

            # Load images
            img_nib = nib.load(str(img_path))
            seg_nib = nib.load(str(s1_seg_path))
            seg_arr = np.array(seg_nib.dataobj)

            # Proper voxel spacing from affine (handles oblique orientations)
            spacing = np.sqrt(np.sum(img_nib.affine[:3,:3]**2, axis=0))
            bbox = get_spine_bbox(seg_arr, pad_mm=args.pad_mm, spacing_xyz=spacing)
            if bbox is None:
                print(f"  SKIP {case_name} — empty Stage1 segmentation")
                continue

            lo, hi = bbox

            # Crop and save for Stage 2
            cropped = crop_nib(img_nib, lo, hi)
            out = s2_input / f"{case_name}_0000.nii.gz"
            nib.save(cropped, str(out))

            bbox_map[case_name] = (img_path, lo, hi)

        print(f"  Cropped {len(bbox_map)} cases for Stage 2")

        # ── Stage 2: L1-L5 classification ─────────────────────────────────────
        print(f"\n[Stage2] L1-L5 classification...")
        s2_output.mkdir()
        run_nnunet_inference(
            s2_input, s2_output,
            dataset_id=args.stage2_dataset,
            fold=args.fold,
            trainer=args.stage2_trainer,
            disable_tta=args.disable_tta,
        )

        # ── Restore to full volume ─────────────────────────────────────────────
        print(f"\n[Restore] Restoring segmentations to original image space...")
        restored = 0
        for case_name, (img_path, lo, hi) in bbox_map.items():
            seg2_path = s2_output / f"{case_name}.nii.gz"
            if not seg2_path.exists():
                print(f"  SKIP {case_name} — no Stage2 output")
                continue

            img_nib  = nib.load(str(img_path))
            seg2_nib = nib.load(str(seg2_path))
            full_seg  = restore_to_full(seg2_nib, img_nib, lo, hi)

            out_path = output_dir / f"{case_name}.nii.gz"
            nib.save(full_seg, str(out_path))
            restored += 1

        print(f"\n✅ Cascade inference complete!")
        print(f"   {restored}/{len(images)} cases processed")
        print(f"   Results: {output_dir}")


if __name__ == "__main__":
    main()
