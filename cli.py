#!/usr/bin/env python3
"""Command-line interface for the lumbar vertebral body segmentation app.

Examples
--------
Single case, GPU:
    python cli.py /data/case001_T1FLAIR.nii.gz \\
        --model-dir $nnUNet_results \\
        --output ./out_case001

Single case on Apple Silicon (M1 Max):
    python cli.py case.nii.gz -m /path/to/nnUNet_results -o ./out --device mps

Batch mode (one folder of inputs):
    python cli.py /data/flair_volumes/ \\
        --model-dir $nnUNet_results \\
        --output ./batch_out
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List

# Self-bootstrap: make the spine_seg package next to this file importable
# regardless of the current working directory.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from spine_seg.config import ModelConfig, PostProcessConfig
from spine_seg.pipeline import run_pipeline


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Segment L1-L5 lumbar vertebral bodies from sagittal T1 FLAIR MRI.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "input",
        type=Path,
        help="Input NIfTI file (.nii or .nii.gz) OR a directory of NIfTI files.",
    )
    p.add_argument(
        "-m", "--model-dir",
        type=Path,
        required=True,
        help="Path to your nnUNet_results directory (parent of Dataset100_SpineL1L5/).",
    )
    p.add_argument(
        "-o", "--output",
        type=Path,
        required=True,
        help="Output directory. For batch mode, one subdir per case is created here.",
    )
    p.add_argument(
        "--folds",
        type=int,
        nargs="+",
        default=None,
        help="Folds to ensemble (default: 0 1 2 3 4). Pass a single fold for fast preview.",
    )
    p.add_argument(
        "--device",
        choices=("cuda", "mps", "cpu"),
        default=None,
        help="Force a device. Default autodetects (cuda > mps > cpu).",
    )
    p.add_argument(
        "--coord-system",
        choices=("LPS", "RAS"),
        default="LPS",
        help="Coordinate system for STL files. LPS matches Slicer's default STL convention.",
    )
    p.add_argument(
        "--smoothing",
        type=int,
        default=10,
        help="Taubin smoothing iterations applied to each STL mesh. Set to 0 to disable.",
    )
    p.add_argument(
        "--no-combined-stl",
        action="store_true",
        help="Skip writing the combined all_vertebrae.stl file.",
    )
    p.add_argument(
        "--drop-raw",
        action="store_true",
        help="Delete the raw (pre-postprocessing) prediction after exporting.",
    )
    p.add_argument(
        "--sagittal-fraction",
        type=float,
        default=PostProcessConfig.sagittal_strip_fraction,
        help="Half-width of the sagittal strip corridor as a fraction of the LR axis.",
    )
    p.add_argument(
        "--min-component-voxels",
        type=int,
        default=PostProcessConfig.min_component_voxels,
        help="Minimum voxels for a label's largest component to be retained.",
    )
    p.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    return p.parse_args(argv)


def _gather_inputs(target: Path) -> List[Path]:
    if target.is_file():
        return [target]
    if target.is_dir():
        cases = sorted(
            list(target.glob("*.nii.gz")) + list(target.glob("*.nii"))
        )
        if not cases:
            raise SystemExit(f"No .nii / .nii.gz files in {target}")
        return cases
    raise SystemExit(f"Input does not exist: {target}")


def main(argv=None) -> int:
    args = _parse_args(argv)
    _setup_logging(args.verbose)
    log = logging.getLogger("spine-seg")

    cases = _gather_inputs(args.input)
    is_batch = len(cases) > 1 or args.input.is_dir()
    args.output.mkdir(parents=True, exist_ok=True)

    pp_cfg = PostProcessConfig(
        sagittal_strip_fraction=args.sagittal_fraction,
        min_component_voxels=args.min_component_voxels,
    )
    model_cfg = ModelConfig()

    failures = 0
    for i, case in enumerate(cases, start=1):
        case_out = args.output / case.stem.replace(".nii", "") if is_batch else args.output
        log.info("[%d/%d] %s -> %s", i, len(cases), case.name, case_out)
        try:
            result = run_pipeline(
                input_nifti=case,
                model_root=args.model_dir,
                output_dir=case_out,
                model_cfg=model_cfg,
                pp_cfg=pp_cfg,
                coord_system=args.coord_system,
                smoothing_iterations=args.smoothing,
                write_combined_stl=not args.no_combined_stl,
                keep_raw_prediction=not args.drop_raw,
                device_pref=args.device,
                folds=args.folds,
            )
            log.info("Voxel counts per label: %s", result.label_voxel_counts)
            log.info("STLs: %s", {k: v.name for k, v in result.stl_per_label.items()})
        except Exception as exc:  # pragma: no cover
            log.exception("Pipeline failed for %s: %s", case.name, exc)
            failures += 1

    if failures:
        log.error("%d/%d cases failed", failures, len(cases))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
