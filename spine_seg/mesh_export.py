"""Marching-cubes STL export, with affine -> world coordinate transform.

Per-vertebra STLs (L1..L5) and an optional combined all_vertebrae.stl, in
either LPS (Slicer's STL convention) or RAS (NIfTI's convention).

Transform chain:
    voxel index -> RAS world (via NIfTI affine)
    -> LPS world (negate X and Y) if coord_system == "LPS"
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import trimesh
from skimage import measure

from .config import LABELS, VERTEBRA_LABELS

log = logging.getLogger(__name__)

_RAS_TO_LPS = np.diag([-1.0, -1.0, 1.0, 1.0])


def _ras_to_world(affine: np.ndarray, coord_system: str) -> np.ndarray:
    """Combine NIfTI affine (RAS) with optional RAS->LPS flip."""
    if coord_system.upper() == "LPS":
        return _RAS_TO_LPS @ affine
    if coord_system.upper() == "RAS":
        return affine
    raise ValueError(f"coord_system must be 'LPS' or 'RAS', got {coord_system!r}")


def _apply_affine(verts: np.ndarray, affine: np.ndarray) -> np.ndarray:
    """Apply 4x4 affine to Nx3 vertex array."""
    homo = np.column_stack([verts, np.ones(len(verts))])
    return (affine @ homo.T).T[:, :3]


def _label_to_mesh(
    seg: np.ndarray,
    label: int,
    affine: np.ndarray,
    *,
    smoothing_iterations: int,
    coord_system: str,
) -> Optional[trimesh.Trimesh]:
    """Build a smoothed surface mesh for a single label, in world coords.

    Returns None if the label is empty.
    """
    mask = seg == label
    if not mask.any():
        return None
    # Pad by 1 so marching-cubes never produces an open boundary at the volume edge.
    padded = np.pad(mask.astype(np.uint8), 1, mode="constant", constant_values=0)
    try:
        verts, faces, normals, _ = measure.marching_cubes(
            padded, level=0.5, allow_degenerate=False,
        )
    except (RuntimeError, ValueError) as e:
        log.warning("marching_cubes failed for label %d: %s", label, e)
        return None
    # Undo the +1 pad offset before applying the affine (which expects voxel indices).
    verts = verts - 1.0
    world_affine = _ras_to_world(affine, coord_system)
    verts_world = _apply_affine(verts, world_affine)
    mesh = trimesh.Trimesh(vertices=verts_world, faces=faces, process=False)
    if smoothing_iterations and smoothing_iterations > 0:
        # Taubin smoothing preserves volume better than Laplacian.
        try:
            trimesh.smoothing.filter_taubin(mesh, iterations=int(smoothing_iterations))
        except Exception as e:
            log.debug("Taubin smoothing skipped: %s", e)
    # Recompute normals after smoothing.
    mesh.fix_normals()
    return mesh


def export_label_stls(
    seg: np.ndarray,
    affine: np.ndarray,
    out_dir: Path,
    *,
    smoothing_iterations: int = 10,
    coord_system: str = "LPS",
) -> Dict[int, Path]:
    """Write one STL per vertebra label present in seg.

    Returns {label_int: Path}. Empty labels are skipped.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[int, Path] = {}
    for lbl in VERTEBRA_LABELS:
        mesh = _label_to_mesh(
            seg, lbl, affine,
            smoothing_iterations=smoothing_iterations,
            coord_system=coord_system,
        )
        if mesh is None or len(mesh.vertices) == 0:
            log.info("Label %s (%d): no voxels, skipping STL.", LABELS.get(lbl, lbl), lbl)
            continue
        out_path = out_dir / f"{LABELS[lbl]}.stl"
        mesh.export(out_path, file_type="stl")
        paths[lbl] = out_path
        log.info("Wrote %s (%d verts, %d faces)", out_path.name, len(mesh.vertices), len(mesh.faces))
    return paths


def export_combined_stl(
    seg: np.ndarray,
    affine: np.ndarray,
    out_path: Path,
    *,
    smoothing_iterations: int = 10,
    coord_system: str = "LPS",
) -> Optional[Path]:
    """Write a single STL containing all 5 vertebra surfaces concatenated."""
    meshes = []
    for lbl in VERTEBRA_LABELS:
        m = _label_to_mesh(
            seg, lbl, affine,
            smoothing_iterations=smoothing_iterations,
            coord_system=coord_system,
        )
        if m is not None and len(m.vertices) > 0:
            meshes.append(m)
    if not meshes:
        return None
    combined = trimesh.util.concatenate(meshes)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.export(out_path, file_type="stl")
    log.info("Wrote combined %s (%d verts)", out_path.name, len(combined.vertices))
    return out_path
