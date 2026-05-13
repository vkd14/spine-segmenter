#!/usr/bin/env python3
"""
Generate 3D surface meshes from NIfTI segmentation masks.

For each label (L1-L5), applies marching cubes + Laplacian smoothing
and exports per-vertebra STL files + a combined coloured PLY file.

The NIfTI affine is applied so physical coordinates (mm space, RAS+) 
are correct — meshes will align perfectly in 3D Slicer / ITK-SNAP / Blender.

Usage:
    # Single case
    python scripts/generate_mesh.py \
        --input  results/seg_cropped/flair_100_20220730_Lumbar_Spine_Sagittal_T1_FLAIR_s6.nii.gz \
        --output results/meshes/flair_100/ \
        --smooth 30

    # Batch (all cases in a directory)
    python scripts/generate_mesh.py \
        --batch \
        --input-dir    results/seg_cropped/ \
        --output-dir   results/meshes/ \
        --case-list    scripts/high_confidence_cases.txt \
        --smooth       30 \
        --jobs         4
"""

import argparse
import numpy as np
import nibabel as nib
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

try:
    from skimage import measure
    from skimage.filters import gaussian
except ImportError:
    raise ImportError("pip install scikit-image")

try:
    import trimesh
except ImportError:
    raise ImportError("pip install trimesh")


# Per-label colours (RGBA 0-255) for coloured PLY
LABEL_COLORS = {
    1: [220,  50,  50, 255],   # L1 — red
    2: [255, 160,  20, 255],   # L2 — orange
    3: [ 50, 200,  50, 255],   # L3 — green
    4: [ 50, 140, 255, 255],   # L4 — blue
    5: [200,  60, 200, 255],   # L5 — purple
}
LABEL_NAMES = {1: 'L1', 2: 'L2', 3: 'L3', 4: 'L4', 5: 'L5'}


def laplacian_smooth(verts: np.ndarray, faces: np.ndarray,
                     iterations: int = 30, factor: float = 0.5) -> np.ndarray:
    """Simple Laplacian smoothing on mesh vertices."""
    v = verts.copy()
    for _ in range(iterations):
        # Build adjacency average
        neighbor_sum = np.zeros_like(v)
        counts = np.zeros(len(v))
        for edge in [(0,1),(1,2),(0,2)]:
            i, j = faces[:, edge[0]], faces[:, edge[1]]
            neighbor_sum[i] += v[j]; counts[i] += 1
            neighbor_sum[j] += v[i]; counts[j] += 1
        avg = neighbor_sum / (counts[:, None] + 1e-8)
        v = v + factor * (avg - v)
    return v


def mesh_label(seg: np.ndarray, label: int, affine: np.ndarray,
               vox_sizes: tuple, smooth_sigma: float = 0.4,
               smooth_iters: int = 30) -> trimesh.Trimesh | None:
    """Extract surface mesh for a single label."""
    mask = (seg == label).astype(np.float32)
    if mask.sum() < 100:
        return None

    # Slight Gaussian blur for smoother surface (avoids staircase)
    if smooth_sigma > 0:
        mask = gaussian(mask, sigma=smooth_sigma, preserve_range=True)

    # Marching cubes (threshold=0.5 on blurred mask)
    try:
        verts, faces, normals, _ = measure.marching_cubes(
            mask, level=0.5, spacing=vox_sizes[:3]
        )
    except (ValueError, RuntimeError):
        return None

    if len(verts) < 10 or len(faces) < 4:
        return None

    # Laplacian smoothing
    verts = laplacian_smooth(verts, faces, iterations=smooth_iters)

    # Apply NIfTI affine transform (voxel → physical RAS mm space)
    # marching_cubes returns verts in mm via spacing, but origin is at (0,0,0)
    # We need to apply the full affine
    vox_origin = affine[:3, 3]
    rot_scale   = affine[:3, :3]
    # verts from marching_cubes are already in mm (via spacing kwarg)
    # but we need to account for direction cosines (rotation/flip in affine)
    # Recompute: convert verts back to voxel space, then apply affine
    verts_vox = verts / np.array(vox_sizes[:3])  # undo spacing → back to voxel
    verts_ras = (rot_scale @ verts_vox.T).T + vox_origin

    mesh = trimesh.Trimesh(vertices=verts_ras, faces=faces, process=False)
    return mesh


def process_one(seg_path: Path, out_dir: Path,
                smooth_iters: int = 30, smooth_sigma: float = 0.4) -> str:
    try:
        seg_nib = nib.load(str(seg_path))
        seg = seg_nib.get_fdata(dtype=np.float32).astype(np.uint8)
        affine = seg_nib.affine
        vox_sizes = tuple(float(v) for v in seg_nib.header.get_zooms()[:3])

        out_dir.mkdir(parents=True, exist_ok=True)
        all_meshes = []
        labels_done = []

        for label in range(1, 6):
            if (seg == label).sum() < 100:
                continue
            mesh = mesh_label(seg, label, affine, vox_sizes,
                              smooth_sigma=smooth_sigma, smooth_iters=smooth_iters)
            if mesh is None:
                continue

            # Export per-vertebra STL
            stl_path = out_dir / f"{LABEL_NAMES[label]}.stl"
            mesh.export(str(stl_path))

            # Add colour for combined PLY
            col = np.array([LABEL_COLORS[label]] * len(mesh.vertices), dtype=np.uint8)
            mesh.visual.vertex_colors = col
            all_meshes.append(mesh)
            labels_done.append(LABEL_NAMES[label])

        if all_meshes:
            combined = trimesh.util.concatenate(all_meshes)
            ply_path = out_dir / "all_vertebrae.ply"
            combined.export(str(ply_path))

        return f"OK: {seg_path.name[:40]} → {labels_done} → {out_dir}"

    except Exception as e:
        return f"ERR: {seg_path.name[:40]}: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",      type=Path, default=None)
    ap.add_argument("--output",     type=Path, default=None)
    ap.add_argument("--batch",      action="store_true")
    ap.add_argument("--input-dir",  type=Path, default=Path("results/seg_cropped"))
    ap.add_argument("--output-dir", type=Path, default=Path("results/meshes"))
    ap.add_argument("--case-list",  type=Path, default=None)
    ap.add_argument("--smooth",     type=int,   default=30, help="Laplacian iterations")
    ap.add_argument("--sigma",      type=float, default=0.4, help="Gaussian blur sigma")
    ap.add_argument("--jobs",       type=int,   default=4)
    args = ap.parse_args()

    if not args.batch:
        out = args.output or args.input.parent / (args.input.name.replace(".nii.gz", "_mesh"))
        print(process_one(args.input, out, args.smooth, args.sigma))
        return

    # Batch
    if args.case_list and args.case_list.exists():
        names = [n.strip() for n in open(args.case_list).readlines() if n.strip()]
        files = [args.input_dir / n for n in names if (args.input_dir / n).exists()]
        print(f"Processing {len(files)} cases from case list")
    else:
        files = sorted(args.input_dir.glob("*.nii.gz"))
        print(f"Processing all {len(files)} cases")

    done = errors = 0
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        futures = {}
        for f in files:
            case_name = f.name.replace(".nii.gz", "")
            out = args.output_dir / case_name
            futures[ex.submit(process_one, f, out, args.smooth, args.sigma)] = f

        for fut in as_completed(futures):
            msg = fut.result()
            (done := done+1) if msg.startswith("OK") else (errors := errors+1)
            if done % 20 == 0 or done <= 3 or msg.startswith("ERR"):
                print(f"  [{done}/{len(files)}] {msg}")

    print(f"\nDone: {done} meshed, {errors} errors")
    print(f"Meshes → {args.output_dir}/")
    print("  Each case: L1.stl L2.stl L3.stl L4.stl L5.stl + all_vertebrae.ply")


if __name__ == "__main__":
    main()
