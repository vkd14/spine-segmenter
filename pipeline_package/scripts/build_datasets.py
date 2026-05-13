#!/usr/bin/env python3
"""
build_datasets.py — Build Dataset200 (binary) and Dataset201 (T1/T1-FLAIR, L1-L5)
for the 2-stage cascaded nnUNet pipeline.

Dataset200: Binary spine (all modalities — SPIDER T1+T2 + T1-FLAIR pseudo)
  - Purpose: Stage 1 localization — locate spine bounding box
  - Labels: {background:0, spine:1}

Dataset201: T1/T1-FLAIR focused, cropped to ROI, L1-L5
  - Purpose: Stage 2 classification — identify each lumbar vertebra
  - Labels: {background:0, L1:1, L2:2, L3:3, L4:4, L5:5}
  - Source: SPIDER T1 + T1-FLAIR pseudo-labels (from flair_cropped if available)
"""

import os
import json
import shutil
import argparse
import numpy as np
import nibabel as nib
import SimpleITK as sitk
from pathlib import Path
from tqdm import tqdm

# ─── Paths ─────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data"
NNUNET_RAW = DATA / "nnUNet_raw"

SPIDER_IMAGES = DATA / "SPIDER" / "images"
SPIDER_MASKS  = DATA / "SPIDER" / "masks"
FLAIR_IMAGES  = DATA / "Sagittal_T1_FLAIR"          # 242 clinical T1-FLAIR (no GT)
FLAIR_PSEUDO  = DATA / "flair_inference_input"       # already nnUNet-formatted
FLAIR_PSEUDO_LABELS = ROOT / "results" / "inference_final_v2"  # existing pseudo-labels

SPIDER_LABEL_TO_L = {201: 1, 202: 2, 203: 3, 204: 4, 205: 5}  # L1-L5


# ─── Helpers ───────────────────────────────────────────────────────────────────

def sitk_to_nib(sitk_img):
    """Convert SimpleITK image to nibabel NIfTI."""
    arr = sitk.GetArrayFromImage(sitk_img)  # zyx
    spacing = sitk_img.GetSpacing()          # xyz
    origin  = sitk_img.GetOrigin()           # xyz
    direction = sitk_img.GetDirection()      # row-major 3x3 xyz->xyz

    # Build affine from ITK direction + spacing + origin
    d = np.array(direction).reshape(3, 3)
    affine = np.eye(4)
    affine[:3, :3] = d * np.array(spacing)
    affine[:3, 3]  = origin

    # ITK array is z,y,x → transpose to x,y,z for nibabel
    arr_xyz = arr.transpose(2, 1, 0)
    return nib.Nifti1Image(arr_xyz, affine)


def save_nib(img, path):
    nib.save(img, str(path))
    return path


def load_sitk_as_nib(path):
    return sitk_to_nib(sitk.ReadImage(str(path)))


def remap_spider_mask_binary(arr):
    """Map SPIDER labels → binary spine (any nonzero → 1)."""
    return (arr > 0).astype(np.int16)


def remap_spider_mask_l1l5(arr):
    """Map SPIDER labels 201-205 → L1-L5 (1-5), rest → 0."""
    out = np.zeros_like(arr, dtype=np.int16)
    for src, dst in SPIDER_LABEL_TO_L.items():
        out[arr == src] = dst
    return out


def make_dataset_json(path, n_train, labels, channel="MRI"):
    d = {
        "channel_names": {"0": channel},
        "labels": labels,
        "numTraining": n_train,
        "file_ending": ".nii.gz",
        "overwrite_image_reader_writer": "SimpleITKIO",
    }
    with open(path / "dataset.json", "w") as f:
        json.dump(d, f, indent=2)
    print(f"  wrote dataset.json → {path}/dataset.json")


def ensure_dirs(ds_path):
    (ds_path / "imagesTr").mkdir(parents=True, exist_ok=True)
    (ds_path / "labelsTr").mkdir(parents=True, exist_ok=True)


# ─── Stage-1: Dataset200 Binary ────────────────────────────────────────────────

def build_dataset200(max_cases=None):
    """All SPIDER T1+T2 → binary spine: spine=1, rest=0."""
    ds_path = NNUNET_RAW / "Dataset200_SpineBinary"
    ensure_dirs(ds_path)

    mha_files = sorted(SPIDER_IMAGES.glob("*.mha"))
    if max_cases:
        mha_files = mha_files[:max_cases]

    count = 0
    print(f"\n[Dataset200] Processing {len(mha_files)} SPIDER images...")
    for img_path in tqdm(mha_files, desc="Dataset200"):
        stem = img_path.stem  # e.g. 100_t1
        mask_path = SPIDER_MASKS / f"{stem}.mha"
        if not mask_path.exists():
            print(f"  SKIP {stem} — no mask")
            continue

        case_id = f"spider_{stem}"

        # Image
        img_nib = load_sitk_as_nib(img_path)
        out_img = ds_path / "imagesTr" / f"{case_id}_0000.nii.gz"
        if not out_img.exists():
            save_nib(img_nib, out_img)

        # Binary mask
        mask_nib = load_sitk_as_nib(mask_path)
        arr_binary = remap_spider_mask_binary(np.array(mask_nib.dataobj))
        bin_nib = nib.Nifti1Image(arr_binary, mask_nib.affine, mask_nib.header)
        out_mask = ds_path / "labelsTr" / f"{case_id}.nii.gz"
        if not out_mask.exists():
            save_nib(bin_nib, out_mask)

        count += 1

    make_dataset_json(
        ds_path, count,
        labels={"background": 0, "spine": 1},
    )
    print(f"[Dataset200] Done. {count} cases → {ds_path}")
    return count


# ─── Stage-2: Dataset201 T1-Only L1-L5 ────────────────────────────────────────

def build_dataset201(use_flair_pseudo=True, max_cases=None):
    """
    SPIDER T1 (GT labels 201-205 → 1-5) + optionally best T1-FLAIR pseudo-labels.
    Images are stored full-volume; ROI cropping is done at inference time by Stage 1.
    """
    ds_path = NNUNET_RAW / "Dataset201_SpineL1L5_T1"
    ensure_dirs(ds_path)

    count = 0

    # ── SPIDER T1 ──
    t1_files = sorted(SPIDER_IMAGES.glob("*_t1.mha"))
    if max_cases:
        t1_files = t1_files[:max_cases]

    print(f"\n[Dataset201] Processing {len(t1_files)} SPIDER T1 images...")
    for img_path in tqdm(t1_files, desc="Dataset201-T1"):
        stem = img_path.stem  # e.g. 100_t1
        mask_path = SPIDER_MASKS / f"{stem}.mha"
        if not mask_path.exists():
            continue

        case_id = f"spider_t1_{stem}"

        # Image
        img_nib = load_sitk_as_nib(img_path)
        out_img = ds_path / "imagesTr" / f"{case_id}_0000.nii.gz"
        if not out_img.exists():
            save_nib(img_nib, out_img)

        # L1-L5 mask (only 201-205 survive, rest → 0)
        mask_nib = load_sitk_as_nib(mask_path)
        arr_raw = np.array(mask_nib.dataobj)
        arr_l = remap_spider_mask_l1l5(arr_raw)

        # Skip if no L1-L5 at all
        if arr_l.max() == 0:
            print(f"  SKIP {stem} — no L1-L5 vertebrae in mask")
            continue

        l_nib = nib.Nifti1Image(arr_l, mask_nib.affine, mask_nib.header)
        out_mask = ds_path / "labelsTr" / f"{case_id}.nii.gz"
        if not out_mask.exists():
            save_nib(l_nib, out_mask)

        count += 1

    # ── T1-FLAIR pseudo-labels ──
    if use_flair_pseudo and FLAIR_PSEUDO_LABELS.exists():
        pseudo_files = sorted(FLAIR_PSEUDO_LABELS.glob("*.nii.gz"))
        print(f"\n[Dataset201] Adding {len(pseudo_files)} T1-FLAIR pseudo-labels...")
        for pseg_path in tqdm(pseudo_files, desc="Dataset201-FLAIR"):
            # Corresponding image in flair_inference_input
            fname = pseg_path.stem.replace(".nii", "")  # strip .nii if leftover
            img_src = FLAIR_PSEUDO / f"{fname}_0000.nii.gz"
            if not img_src.exists():
                continue

            case_id = f"flair_{fname}"

            # Check pseudo-label quality: need >=4 L-levels present
            seg = nib.load(str(pseg_path))
            arr = np.array(seg.dataobj, dtype=np.int16)
            n_labels = len([l for l in [1,2,3,4,5] if (arr==l).any()])
            if n_labels < 4:
                continue  # skip poor pseudo-labels

            out_img = ds_path / "imagesTr" / f"{case_id}_0000.nii.gz"
            if not out_img.exists():
                shutil.copy2(img_src, out_img)

            out_mask = ds_path / "labelsTr" / f"{case_id}.nii.gz"
            if not out_mask.exists():
                shutil.copy2(pseg_path, out_mask)

            count += 1

    make_dataset_json(
        ds_path, count,
        labels={"background": 0, "L1": 1, "L2": 2, "L3": 3, "L4": 4, "L5": 5},
    )
    print(f"[Dataset201] Done. {count} cases → {ds_path}")
    return count


# ─── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["200", "201", "all"], default="all",
                        help="Which dataset to build (200=binary, 201=T1 L1-L5, all=both)")
    parser.add_argument("--no-flair-pseudo", action="store_true",
                        help="Don't add FLAIR pseudo-labels to Dataset201")
    parser.add_argument("--max-cases", type=int, default=None,
                        help="Limit cases (for testing)")
    args = parser.parse_args()

    if args.stage in ("200", "all"):
        n200 = build_dataset200(max_cases=args.max_cases)

    if args.stage in ("201", "all"):
        n201 = build_dataset201(
            use_flair_pseudo=not args.no_flair_pseudo,
            max_cases=args.max_cases,
        )

    print("\n✅ Done!")
    if args.stage in ("200", "all"):
        print(f"  Dataset200 (binary):    {n200:3d} cases")
    if args.stage in ("201", "all"):
        print(f"  Dataset201 (T1 L1-L5): {n201:3d} cases")
