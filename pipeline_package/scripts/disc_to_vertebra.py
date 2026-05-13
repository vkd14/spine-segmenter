#!/usr/bin/env python3
"""
disc_to_vertebra.py — Convert disc segmentations to vertebral body segmentations.

Given ensemble disc predictions (labels L1-L5), derive vertebral body masks
by filling the space between consecutive discs along the spine corridor,
using per-disc AP extent as the height constraint for each body.

Usage:
    python scripts/disc_to_vertebra.py \\
        --input  results/ensemble_flair_postprocessed/ \\
        --output results/ensemble_vertebra_bodies/
"""

import argparse
import numpy as np
import nibabel as nib
from pathlib import Path

VERTEBRA_LABELS = [1, 2, 3, 4, 5]
LABEL_NAMES = {1: "L1", 2: "L2", 3: "L3", 4: "L4", 5: "L5"}


def disc_to_vertebra_body(seg):
    """
    Convert 3D disc segmentation (labels 1-5) to vertebral body segmentation.
    
    Each vertebral body N fills the gap between disc N-1 and disc N,
    constrained to the AP/LR extent of the neighboring discs.
    """
    seg = seg.copy().astype(int)
    shape = seg.shape
    
    fg = np.array(np.where(seg > 0)).T
    if len(fg) == 0:
        return seg
    
    spans = fg.max(axis=0) - fg.min(axis=0)
    lr_axis = int(np.argmin(spans))   # narrowest = left-right
    si_axis = int(np.argmax(spans))   # widest = along spine
    ap_axis = 3 - lr_axis - si_axis   # remaining = anterior-posterior
    
    # Collect disc info
    disc_info = {}
    for lab in VERTEBRA_LABELS:
        mask = seg == lab
        if not mask.any():
            continue
        coords = np.array(np.where(mask)).T
        disc_info[lab] = {
            'centroid_si': coords[:, si_axis].mean(),
            'si_min': int(coords[:, si_axis].min()),
            'si_max': int(coords[:, si_axis].max()),
            'ap_min': int(coords[:, ap_axis].min()),
            'ap_max': int(coords[:, ap_axis].max()),
            'lr_min': int(coords[:, lr_axis].min()),
            'lr_max': int(coords[:, lr_axis].max()),
        }
    
    if len(disc_info) < 2:
        return seg
    
    # Sort discs by SI position
    present = sorted(disc_info.keys(), key=lambda l: disc_info[l]['centroid_si'])
    ascending = disc_info[present[0]]['centroid_si'] < disc_info[present[-1]]['centroid_si']
    
    # Compute avg disc-to-disc gap
    gaps = []
    for i in range(len(present) - 1):
        d1 = disc_info[present[i]]
        d2 = disc_info[present[i+1]]
        gaps.append(abs(d2['centroid_si'] - d1['centroid_si']))
    avg_gap = np.mean(gaps) if gaps else 20
    
    vert_seg = np.zeros(shape, dtype=int)
    
    for body_idx, lab in enumerate(present):
        d = disc_info[lab]
        
        # --- SI bounds ---
        if body_idx == 0:
            # First body: extend before first disc by ~avg_gap
            if ascending:
                si_lo = max(0, d['si_min'] - int(avg_gap * 0.7))
                si_hi = d['si_min'] - 1
            else:
                si_lo = d['si_max'] + 1
                si_hi = min(shape[si_axis] - 1, d['si_max'] + int(avg_gap * 0.7))
        elif body_idx == len(present) - 1:
            # Last body: extend after last disc by ~avg_gap
            prev_d = disc_info[present[body_idx - 1]]
            if ascending:
                si_lo = prev_d['si_max'] + 1
                si_hi = d['si_min'] - 1
            else:
                si_lo = d['si_max'] + 1
                si_hi = prev_d['si_min'] - 1
        else:
            # Middle body: gap between previous disc and current disc
            prev_d = disc_info[present[body_idx - 1]]
            if ascending:
                si_lo = prev_d['si_max'] + 1
                si_hi = d['si_min'] - 1
            else:
                si_lo = d['si_max'] + 1
                si_hi = prev_d['si_min'] - 1
        
        if si_lo > si_hi:
            si_lo, si_hi = si_hi, si_lo
        if si_hi - si_lo < 1:
            continue
        
        # --- AP bounds: interpolate between neighboring disc AP ranges ---
        if body_idx == 0:
            ap_lo = d['ap_min']
            ap_hi = d['ap_max']
        elif body_idx == len(present) - 1:
            prev_d = disc_info[present[body_idx - 1]]
            ap_lo = min(prev_d['ap_min'], d['ap_min'])
            ap_hi = max(prev_d['ap_max'], d['ap_max'])
        else:
            prev_d = disc_info[present[body_idx - 1]]
            ap_lo = min(prev_d['ap_min'], d['ap_min'])
            ap_hi = max(prev_d['ap_max'], d['ap_max'])
        
        # Small AP padding
        ap_pad = max(2, int((ap_hi - ap_lo) * 0.08))
        ap_lo = max(0, ap_lo - ap_pad)
        ap_hi = min(shape[ap_axis] - 1, ap_hi + ap_pad)
        
        # --- LR bounds: use disc LR extent ---
        if body_idx > 0:
            prev_d = disc_info[present[body_idx - 1]]
            lr_lo = min(prev_d['lr_min'], d['lr_min'])
            lr_hi = max(prev_d['lr_max'], d['lr_max'])
        else:
            lr_lo = d['lr_min']
            lr_hi = d['lr_max']
        
        lr_pad = max(1, int((lr_hi - lr_lo) * 0.1))
        lr_lo = max(0, lr_lo - lr_pad)
        lr_hi = min(shape[lr_axis] - 1, lr_hi + lr_pad)
        
        # Fill the body region
        body_slices = [slice(None)] * 3
        body_slices[si_axis] = slice(si_lo, si_hi + 1)
        body_slices[ap_axis] = slice(ap_lo, ap_hi + 1)
        body_slices[lr_axis] = slice(lr_lo, lr_hi + 1)
        
        body_mask = np.zeros(shape, dtype=bool)
        body_mask[tuple(body_slices)] = True
        body_mask[seg > 0] = False         # exclude disc voxels
        body_mask[vert_seg > 0] = False    # no overlap with other bodies
        
        vert_seg[body_mask] = lab
    
    return vert_seg


def process_file(in_path, out_path, verbose=False):
    """Process a single NIfTI file: disc → vertebral body."""
    img = nib.load(str(in_path))
    seg = img.get_fdata(dtype=np.float32).astype(np.int32)
    
    disc_count = {l: (seg == l).sum() for l in VERTEBRA_LABELS}
    vert_body = disc_to_vertebra_body(seg)
    body_count = {l: (vert_body == l).sum() for l in VERTEBRA_LABELS}
    
    if verbose:
        for l in VERTEBRA_LABELS:
            ratio = body_count[l] / disc_count[l] if disc_count[l] > 0 else 0
            print(f"    L{l}: disc={disc_count[l]:>7,} → body={body_count[l]:>7,} ({ratio:.1f}x)")
    
    out_img = nib.Nifti1Image(vert_body.astype(np.uint8), img.affine, img.header)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(out_img, str(out_path))


def main():
    parser = argparse.ArgumentParser(description="Convert disc segmentations to vertebral body segmentations")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--n", type=int, default=0, help="Process only N files (0=all)")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    
    in_dir = Path(args.input)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    files = sorted(in_dir.glob("*.nii.gz"))
    if args.n > 0:
        files = files[:args.n]
    
    print(f"\nDisc → Vertebral Body Conversion")
    print(f"  Input  : {in_dir} ({len(list(in_dir.glob('*.nii.gz')))} files)")
    print(f"  Output : {out_dir}")
    print(f"  Process: {len(files)} files")
    print()
    
    done = 0
    for i, f in enumerate(files):
        out = out_dir / f.name
        try:
            process_file(f, out, verbose=args.verbose or i < 3)
            done += 1
            if (i + 1) % 30 == 0 or i < 3:
                print(f"  [{i+1}/{len(files)}] ✓ {f.name}")
        except Exception as e:
            print(f"  [{i+1}/{len(files)}] ✗ {f.name}: {e}")
    
    print(f"\n✓ Done: {done}/{len(files)} files → {out_dir}")


if __name__ == "__main__":
    main()
