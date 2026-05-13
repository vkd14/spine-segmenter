"""3D Slicer-friendly exports: NIfTI label volume, color table, README."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

import nibabel as nib
import numpy as np

from .config import LABEL_COLORS, LABELS, VERTEBRA_LABELS

log = logging.getLogger(__name__)


def save_segmentation_nifti(
    seg: np.ndarray,
    affine: np.ndarray,
    out_path: Path,
) -> Path:
    """Write a uint8 multi-label NIfTI with the original affine."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img = nib.Nifti1Image(seg.astype(np.uint8), affine)
    img.set_data_dtype(np.uint8)
    nib.save(img, str(out_path))
    log.info("Wrote %s (%s)", out_path, seg.shape)
    return out_path


def write_slicer_color_table(out_path: Path) -> Path:
    """Write a Slicer-compatible .ctbl color table for L1..L5.

    Format (one row per label):
        label_id name R G B A
    R/G/B in [0,255], A in [0,255].
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["0 background 0 0 0 0"]
    for lbl in VERTEBRA_LABELS:
        r, g, b = LABEL_COLORS[lbl]
        lines.append(f"{lbl} {LABELS[lbl]} {r} {g} {b} 255")
    out_path.write_text("\n".join(lines) + "\n")
    return out_path


def write_slicer_readme(out_path: Path, files: Dict[str, str]) -> Path:
    """Write a brief Slicer-load instruction file."""
    out_path = Path(out_path)
    text = (
        "# Loading these outputs into 3D Slicer\n\n"
        "1. Open 3D Slicer.\n"
        "2. Drag the original T1 FLAIR `.nii.gz` -> load as **Volume**.\n"
        "3. Drag `segmentation.nii.gz` -> choose **Segmentation** as the type.\n"
        "   (On Slicer < 5, load as LabelMap then right-click -> Convert labelmap to segmentation.)\n"
        "4. Optional: load `labels.ctbl` via the *Colors* module before converting,\n"
        "   to get the L1..L5 names and colors.\n"
        "5. Drag any `.stl` from `meshes/` -> choose **Model**. Coordinate system is LPS.\n\n"
        "Both the segmentation NIfTI and the STL meshes are in the volume's world\n"
        "coordinates (LPS for STLs, RAS for NIfTI), so they overlay correctly on\n"
        "the original MRI.\n\n"
        "## Files in this bundle\n\n"
    )
    for name, desc in files.items():
        text += f"- `{name}` - {desc}\n"
    out_path.write_text(text)
    return out_path
