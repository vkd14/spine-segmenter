"""nnUNetPredictor wrapper for the lumbar L1-L5 ResEncUNet ensemble.

Wraps the same flags the PDF documents (Section 5.3):
    -d 100 -c 3d_fullres -tr nnUNetTrainer -p nnUNetResEncUNetLPlans
    -f 0 1 2 3 4 -chk checkpoint_best.pth -step_size 0.5

Handles:
  - 1-fold (fast preview, default on M1 Max) or 5-fold ensemble
  - Device autodetect: cuda > mps > cpu (override via device_pref)
  - Naming convention: copies/symlinks input file to <case>_0000.nii.gz
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterable, Optional, Tuple

import torch

from .config import ModelConfig

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# tqdm progress callback — injected into nnU-Net's internal progress bars.
# ---------------------------------------------------------------------------
# nnU-Net uses tqdm.std.tqdm everywhere via `from tqdm import tqdm`. By
# monkeypatching the .update and .close methods on the tqdm CLASS itself,
# every existing and future instance fires our callback — independent of
# import order in the various nnU-Net submodules.

ProgressCallback = Callable[[int, int, str], None]   # (n, total, desc)

_progress_lock = threading.Lock()
_active_callback: Optional[ProgressCallback] = None


def _install_tqdm_hooks() -> None:
    import tqdm.std as _tstd  # local import keeps module load cheap

    if getattr(_tstd.tqdm, "_spineseg_patched", False):
        return  # already patched (idempotent)

    _orig_update = _tstd.tqdm.update
    _orig_close = _tstd.tqdm.close

    def _patched_update(self, n=1):
        result = _orig_update(self, n)
        cb = _active_callback
        if cb is not None:
            total = getattr(self, "total", None)
            if total:
                try:
                    cb(int(self.n), int(total), str(self.desc or ""))
                except Exception:
                    pass  # never let UI hiccups break inference
        return result

    def _patched_close(self):
        cb = _active_callback
        if cb is not None:
            total = getattr(self, "total", None)
            if total:
                try:
                    cb(int(total), int(total), str(self.desc or ""))
                except Exception:
                    pass
        return _orig_close(self)

    _tstd.tqdm.update = _patched_update
    _tstd.tqdm.close = _patched_close
    _tstd.tqdm._spineseg_patched = True


_install_tqdm_hooks()


@contextmanager
def progress_callback_active(callback: Optional[ProgressCallback]):
    """Set the active tqdm callback for the duration of the `with` block."""
    global _active_callback
    with _progress_lock:
        _active_callback = callback
    try:
        yield
    finally:
        with _progress_lock:
            _active_callback = None


def autodetect_device() -> torch.device:
    """Pick the best available device on this machine."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_device(pref: Optional[str]) -> torch.device:
    """Map a user preference string to a torch.device, with sane fallback."""
    if pref is None or pref == "auto":
        return autodetect_device()
    pref = pref.lower()
    if pref == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if pref == "mps" and torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    if pref == "cpu":
        return torch.device("cpu")
    log.warning("Requested device %r unavailable; falling back to autodetect.", pref)
    return autodetect_device()


def _model_folder(model_root: Path, cfg: ModelConfig) -> Path:
    """Build the nnUNet trained-model directory path."""
    trainer_dir = f"{cfg.trainer}__{cfg.plans}__{cfg.configuration}"
    return Path(model_root) / cfg.dataset_name / trainer_dir


def _stage_input(input_nifti: Path, staging_dir: Path) -> Tuple[Path, str]:
    """Copy/symlink input file to nnUNet's <case>_0000.nii.gz convention.

    Returns (staging_dir, case_id).
    """
    name = input_nifti.name
    if name.endswith(".nii.gz"):
        case_id = name[: -len(".nii.gz")]
    elif name.endswith(".nii"):
        case_id = name[: -len(".nii")]
    else:
        case_id = input_nifti.stem
    # Strip a stray trailing _0000 (so we don't end up with foo_0000_0000)
    if case_id.endswith("_0000"):
        case_id = case_id[: -len("_0000")]
    staged = staging_dir / f"{case_id}_0000.nii.gz"
    if staged.exists():
        staged.unlink()
    # nnU-Net needs .nii.gz; if input is uncompressed .nii, just copy.
    # Symlink keeps disk usage low for repeated runs.
    try:
        os.symlink(str(input_nifti.resolve()), str(staged))
    except OSError:
        shutil.copyfile(input_nifti, staged)
    return staged, case_id


def run_inference(
    *,
    input_nifti: Path,
    model_root: Path,
    output_dir: Path,
    cfg: Optional[ModelConfig] = None,
    folds: Optional[Iterable[int]] = None,
    device_pref: Optional[str] = None,
    keep_staging: bool = False,
    progress_callback: Optional[ProgressCallback] = None,
) -> Path:
    """Run nnU-Net (single fold or ensemble) on one NIfTI volume.

    Returns the path to the predicted segmentation NIfTI:
        <output_dir>/<case_id>.nii.gz
    """
    cfg = cfg or ModelConfig()
    if folds is None:
        folds = cfg.folds
    folds = tuple(int(f) for f in folds)

    model_dir = _model_folder(Path(model_root), cfg)
    if not model_dir.is_dir():
        raise FileNotFoundError(
            f"Trained-model directory not found: {model_dir}\n"
            f"Expected layout: <model_root>/{cfg.dataset_name}/"
            f"{cfg.trainer}__{cfg.plans}__{cfg.configuration}/fold_*/{cfg.checkpoint_name}"
        )

    # Verify each requested fold has the checkpoint.
    missing = [f for f in folds if not (model_dir / f"fold_{f}" / cfg.checkpoint_name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing checkpoints for folds {missing} in {model_dir}\n"
            f"(looking for fold_X/{cfg.checkpoint_name})"
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Stage input into nnU-Net's expected naming convention.
    staging_dir = Path(tempfile.mkdtemp(prefix="spine_seg_in_"))
    try:
        _, case_id = _stage_input(Path(input_nifti), staging_dir)

        # Lazy-import nnUNet so the module imports cheaply when the user just
        # wants post-processing / mesh helpers.
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

        device = resolve_device(device_pref)
        log.info(
            "nnU-Net inference: folds=%s device=%s model=%s",
            folds, device, model_dir,
        )

        predictor = nnUNetPredictor(
            tile_step_size=cfg.tile_step_size,
            use_gaussian=cfg.use_gaussian,
            use_mirroring=cfg.use_mirroring,
            perform_everything_on_device=device.type != "cpu",
            device=device,
            verbose=False,
            verbose_preprocessing=False,
            allow_tqdm=True,
        )
        predictor.initialize_from_trained_model_folder(
            str(model_dir),
            use_folds=folds,
            checkpoint_name=cfg.checkpoint_name,
        )

        # On macOS, multiprocessing for the segmentation export can hang due to
        # MPS fork limitations. Run single-process for safety.
        n_proc = 0 if device.type == "mps" else 2
        with progress_callback_active(progress_callback):
            predictor.predict_from_files(
                list_of_lists_or_source_folder=str(staging_dir),
                output_folder_or_list_of_truncated_output_files=str(output_dir),
                save_probabilities=False,
                overwrite=True,
                num_processes_preprocessing=max(1, n_proc),
                num_processes_segmentation_export=max(1, n_proc),
            )

        pred_path = output_dir / f"{case_id}.nii.gz"
        if not pred_path.exists():
            # nnU-Net may have written some other extension; pick first .nii.gz.
            candidates = sorted(output_dir.glob("*.nii.gz"))
            if not candidates:
                raise RuntimeError(f"nnUNetPredictor produced no output in {output_dir}")
            pred_path = candidates[0]
        log.info("Inference output: %s", pred_path)
        return pred_path
    finally:
        if not keep_staging:
            shutil.rmtree(staging_dir, ignore_errors=True)
