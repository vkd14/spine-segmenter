#!/usr/bin/env python3
"""
Filter pseudo-labels from T1-FLAIR inference to keep top 100-150 high-confidence cases.

Scoring criteria:
  1. Number of labels present (≥4/5 required, 5/5 scores highest)
  2. Volume balance — each vertebra voxel count within reasonable range
  3. Vertical ordering — L1 centroid superior to L5
  4. Post-processing removal rate — cases where <35% voxels were removed

Usage:
    python scripts/filter_pseudolabels.py \
        --raw   results/inference_raw_v2/ \
        --final results/inference_final_v2/ \
        --output filtered_cases.txt \
        --top 120
"""

import argparse, csv
import numpy as np
import nibabel as nib
from pathlib import Path


def score_case(raw_f: Path, final_f: Path) -> dict:
    """Score a pseudo-label case. Returns dict with score and diagnostics."""
    try:
        seg_raw   = nib.load(raw_f).get_fdata(dtype=np.float32).astype(np.uint8)
        seg_final = nib.load(final_f).get_fdata(dtype=np.float32).astype(np.uint8) if final_f.exists() else seg_raw
    except Exception as e:
        return {"score": -1, "error": str(e)}

    score = 0.0
    info = {}

    # --- 1. Label presence (0-50 pts) ---
    voxels = {l: int((seg_final == l).sum()) for l in range(1, 6)}
    present = [l for l, v in voxels.items() if v > 300]
    info["n_labels"] = len(present)
    info["voxels"] = voxels
    score += len(present) * 10  # 10 pts per label found

    if len(present) < 4:
        return {"score": score, **info}  # hard reject

    # --- 2. Volume balance (0-20 pts) ---
    vols = [voxels[l] for l in present]
    if len(vols) > 1:
        cv = np.std(vols) / (np.mean(vols) + 1e-6)
        # lower CV = more balanced vertebrae
        balance_score = max(0, 20 - cv * 40)
        score += balance_score
        info["vol_cv"] = round(float(cv), 3)

    # --- 3. Anatomical ordering (0-20 pts) ---
    centroids = {}
    for l in present:
        coords = np.argwhere(seg_final == l)
        centroids[l] = float(coords[:, 2].mean())  # Z centroid (superior-inferior)
    ordered = all(centroids.get(l, 0) > centroids.get(l+1, -1)
                  for l in present if l+1 in centroids)
    info["ordered"] = ordered
    if ordered:
        score += 20

    # --- 4. Post-processing stability (0-10 pts) ---
    raw_total   = int((seg_raw > 0).sum())
    final_total = int((seg_final > 0).sum())
    if raw_total > 0:
        removed_frac = 1 - final_total / raw_total
        info["removed_frac"] = round(removed_frac, 3)
        if removed_frac < 0.15:
            score += 10   # very clean
        elif removed_frac < 0.35:
            score += 5    # acceptable
        # >35% removed → fragmented prediction, no bonus

    info["score"] = round(score, 2)
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw",    default="results/inference_raw_v2")
    ap.add_argument("--final",  default="results/inference_final_v2")
    ap.add_argument("--output", default="scripts/high_confidence_cases.txt")
    ap.add_argument("--top",    type=int, default=120)
    args = ap.parse_args()

    raw_dir   = Path(args.raw)
    final_dir = Path(args.final)
    out_path  = Path(args.output)

    raw_files = sorted(raw_dir.glob("*.nii.gz"))
    print(f"Scoring {len(raw_files)} cases...")

    results = []
    for i, rf in enumerate(raw_files):
        ff = final_dir / rf.name
        info = score_case(rf, ff)
        info["name"] = rf.name
        results.append(info)
        if (i+1) % 20 == 0:
            print(f"  [{i+1}/{len(raw_files)}] latest score: {info.get('score', 0):.1f}")

    # Sort by score descending
    results.sort(key=lambda x: x.get("score", -1), reverse=True)

    top_cases = [r for r in results if r.get("score", 0) >= 30][:args.top]
    print(f"\nTop {len(top_cases)} cases (score ≥30):")
    for r in top_cases[:10]:
        print(f"  {r['name'][:45]}: score={r.get('score')}, labels={r.get('n_labels')}, "
              f"removed={r.get('removed_frac','?')}, ordered={r.get('ordered','?')}")

    # Write output list
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        for r in top_cases:
            f.write(r["name"] + "\n")

    # Also write full CSV for inspection
    csv_path = out_path.with_suffix(".csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["name","score","n_labels","vol_cv","ordered","removed_frac","error"])
        w.writeheader()
        for r in results:
            w.writerow({k: r.get(k, "") for k in w.fieldnames})

    print(f"\nSaved {len(top_cases)} cases → {out_path}")
    print(f"Full scores → {csv_path}")


if __name__ == "__main__":
    main()
