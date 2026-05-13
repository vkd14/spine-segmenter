"""End-to-end pipeline orchestrator.

Stage 1: nnU-Net (1-fold default, 5-fold optional ensemble)
Stage 2: 3-step anatomical post-processing (PDF Section 5.4)
Stage 3: NIfTI export (uint8, original affine)
Stage 4: STL extraction per label (marching cubes + Taubin smoothing)
Stage 5: Slicer color table + README
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional

import nibabel as nib
import numpy as np

from .config import LABELS, ModelConfig, PostProcessConfig
from .inference import ProgressCallback, run_inference
from .mesh_export import export_combined_stl, export_label_stls
from .postprocess import label_summary, postprocess
from .slicer_export import save_segmentation_nifti, write_slicer_color_table, write_slicer_readme

log = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Everything a caller might want after a successful run."""

    case_id: str
    output_dir: Path
    raw_prediction: Optional[Path]
    cleaned_segmentation: Path
    color_table: Path
    slicer_readme: Path
    stl_per_label: Dict[int, Path] = field(default_factory=dict)
    combined_stl: Optional[Path] = None
    label_voxel_counts: Dict[int, int] = field(default_factory=dict)
    folds_used: tuple = ()
    device: str = ""


def _case_id_from_path(p: Path) -> str:
    name = p.name
    if name.endswith(".nii.gz"):
        return name[: -len(".nii.gz")]
    if name.endswith(".nii"):
        return name[: -len(".nii")]
    return p.stem


def run_pipeline(
    *,
    input_nifti: Path | str,
    model_root: Path | str,
    output_dir: Path | str,
    model_cfg: Optional[ModelConfig] = None,
    pp_cfg: Optional[PostProcessConfig] = None,
    coord_system: str = "LPS",
    smoothing_iterations: int = 10,
    write_combined_stl: bool = True,
    keep_raw_prediction: bool = True,
    device_pref: Optional[str] = None,
    folds: Optional[Iterable[int]] = None,
    progress_callback: Optional[ProgressCallback] = None,
    stage_callback: Optional[Callable[[str, float], None]] = None,
) -> PipelineResult:
    """Process one volume end-to-end. See README for output layout."""

    input_nifti = Path(input_nifti)
    model_root = Path(model_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_cfg = model_cfg or ModelConfig()
    pp_cfg = pp_cfg or PostProcessConfig()

    case_id = _case_id_from_path(input_nifti)
    log.info("=== %s -> %s ===", input_nifti.name, output_dir)

    def _stage(name: str, frac: float) -> None:
        if stage_callback is not None:
            try:
                stage_callback(name, frac)
            except Exception:
                pass

    # --- Stage 1: nnU-Net inference ----------------------------------------
    _stage("Running nnU-Net inference", 0.02)
    raw_dir = output_dir / "raw_prediction"
    raw_dir.mkdir(parents=True, exist_ok=True)
    pred_path = run_inference(
        input_nifti=input_nifti,
        model_root=model_root,
        output_dir=raw_dir,
        cfg=model_cfg,
        folds=folds,
        device_pref=device_pref,
        progress_callback=progress_callback,
    )
    raw_img = nib.load(str(pred_path))
    raw_seg = np.asarray(raw_img.dataobj).astype(np.uint8)
    affine = raw_img.affine

    # --- Stage 2: Post-processing ------------------------------------------
    _stage("Anatomical post-processing", 0.85)
    log.info("Post-processing (sagittal strip + largest CC + SI ordering)")
    cleaned = postprocess(raw_seg, cfg=pp_cfg)
    counts = label_summary(cleaned)
    log.info("Voxel counts after cleanup: %s", counts)

    # --- Stage 3: NIfTI export ---------------------------------------------
    _stage("Saving segmentation NIfTI", 0.90)
    seg_path = save_segmentation_nifti(cleaned, affine, output_dir / "segmentation.nii.gz")

    # --- Stage 4: STL meshes -----------------------------------------------
    _stage("Generating STL meshes", 0.92)
    meshes_dir = output_dir / "meshes"
    stl_paths = export_label_stls(
        cleaned, affine, meshes_dir,
        smoothing_iterations=smoothing_iterations,
        coord_system=coord_system,
    )
    combined_path = None
    if write_combined_stl:
        combined_path = export_combined_stl(
            cleaned, affine, meshes_dir / "all_vertebrae.stl",
            smoothing_iterations=smoothing_iterations,
            coord_system=coord_system,
        )

    # --- Stage 5: Slicer helpers -------------------------------------------
    _stage("Writing Slicer helpers", 0.99)
    ctbl_path = write_slicer_color_table(output_dir / "labels.ctbl")
    readme_files = {
        "segmentation.nii.gz": "Cleaned multi-label NIfTI (uint8, L1=1..L5=5).",
        "labels.ctbl": "Slicer color table (load via Colors module).",
        "meshes/L?.stl": "Per-vertebra surface meshes in LPS world coords.",
        "raw_prediction/<case>.nii.gz": "Pre-postprocessing nnU-Net output (kept for QA).",
    }
    readme_path = write_slicer_readme(output_dir / "slicer_README.md", readme_files)

    # --- Optional: drop raw prediction --------------------------------------
    raw_kept: Optional[Path] = pred_path
    if not keep_raw_prediction:
        try:
            shutil.rmtree(raw_dir, ignore_errors=True)
            raw_kept = None
        except Exception:
            log.warning("Could not delete raw_prediction directory.")

    folds_used = tuple(folds) if folds is not None else model_cfg.folds
    from .inference import resolve_device  # local import to avoid cycles
    device = str(resolve_device(device_pref))

    return PipelineResult(
        case_id=case_id,
        output_dir=output_dir,
        raw_prediction=raw_kept,
        cleaned_segmentation=seg_path,
        color_table=ctbl_path,
        slicer_readme=readme_path,
        stl_per_label=stl_paths,
        combined_stl=combined_path,
        label_voxel_counts=counts,
        folds_used=folds_used,
        device=device,
    )
