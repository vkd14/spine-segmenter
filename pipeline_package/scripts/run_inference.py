"""
Inference Script — Run trained nnU-Net on new MRI volumes.

Produces per-vertebra L1-L5 masks and color-coded overlays.
Optionally ensembles with V7 MPS model for better robustness.
"""

import os
import json
import logging
import argparse
import subprocess
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    import nibabel as nib
except ImportError:
    raise ImportError("nibabel required")

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import yaml
except ImportError:
    raise ImportError("PyYAML required")


VERTEBRA_COLORS = {
    1: (0, 255, 0),     # L1 = green
    2: (255, 128, 0),   # L2 = blue-orange
    3: (0, 255, 255),   # L3 = yellow
    4: (255, 0, 255),   # L4 = magenta
    5: (0, 128, 255),   # L5 = orange
}

VERTEBRA_NAMES = {1: 'L1', 2: 'L2', 3: 'L3', 4: 'L4', 5: 'L5'}


def run_nnunet_predict(
    input_dir: str,
    output_dir: str,
    dataset_id: int = 100,
    plans: str = "nnUNetResEncUNetLPlans",
    config: str = "3d_fullres",
    trainer: str = "nnUNetTrainer",
    folds: str = "0 1 2 3 4",
):
    """Run nnU-Net prediction with ensemble of all folds."""
    cmd = [
        "nnUNetv2_predict",
        "-i", str(input_dir),
        "-o", str(output_dir),
        "-d", str(dataset_id),
        "-c", config,
        "-tr", trainer,
        "-p", plans,
        "-f", *folds.split(),
    ]

    logger.info(f"Running nnU-Net inference on {input_dir}")
    subprocess.run(cmd, check=True)


def create_per_vertebra_outputs(
    prediction_dir: str,
    input_dir: str,
    output_dir: str,
):
    """Create per-vertebra masks and overlays from nnU-Net predictions."""
    pred_path = Path(prediction_dir)
    in_path = Path(input_dir)
    out_path = Path(output_dir)
    overlay_dir = out_path / "overlays"
    mask_dir = out_path / "per_vertebra_masks"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    pred_files = sorted(pred_path.glob("*.nii.gz"))
    results = {}

    for pf in pred_files:
        case_id = pf.stem.replace('.nii', '')

        # Load prediction
        pred_nii = nib.load(str(pf))
        pred = pred_nii.get_fdata().astype(np.int32)

        # Load original image for overlay
        img_file = in_path / f"{case_id}_0000.nii.gz"
        if not img_file.exists():
            img_file = in_path / f"{case_id}.nii.gz"
        if img_file.exists():
            img = nib.load(str(img_file)).get_fdata().astype(np.float32)
        else:
            img = None

        # Extract per-vertebra masks
        case_results = {}
        for label_val, label_name in VERTEBRA_NAMES.items():
            vert_mask = (pred == label_val).astype(np.uint8)
            if vert_mask.sum() > 0:
                # Save 3D mask
                mask_nii = nib.Nifti1Image(vert_mask, pred_nii.affine)
                nib.save(mask_nii, str(mask_dir / f"{case_id}_{label_name}.nii.gz"))

                # Compute stats
                coords = np.where(vert_mask > 0)
                case_results[label_name] = {
                    'volume_voxels': int(vert_mask.sum()),
                    'centroid': [int(np.mean(c)) for c in coords],
                }

        results[case_id] = case_results

        # Create center-slice overlay
        if img is not None and cv2 is not None and pred.ndim == 3:
            # Get center sagittal slice
            sag_axis = np.argmin(pred.shape)
            center_idx = pred.shape[sag_axis] // 2

            if sag_axis == 0:
                img_slice = img[center_idx]
                pred_slice = pred[center_idx]
            elif sag_axis == 1:
                img_slice = img[:, center_idx]
                pred_slice = pred[:, center_idx]
            else:
                img_slice = img[:, :, center_idx]
                pred_slice = pred[:, :, center_idx]

            # Normalize image
            smin, smax = img_slice.min(), img_slice.max()
            if smax - smin > 1e-6:
                normalized = ((img_slice - smin) / (smax - smin) * 255).astype(np.uint8)
            else:
                normalized = np.zeros_like(img_slice, dtype=np.uint8)

            overlay = cv2.cvtColor(normalized, cv2.COLOR_GRAY2BGR)

            for label_val, color in VERTEBRA_COLORS.items():
                vert_mask_2d = (pred_slice == label_val).astype(np.uint8)
                if vert_mask_2d.sum() == 0:
                    continue

                color_mask = np.zeros_like(overlay)
                color_mask[vert_mask_2d > 0] = color
                overlay = cv2.addWeighted(overlay, 1.0, color_mask, 0.35, 0)

                contours, _ = cv2.findContours(vert_mask_2d, cv2.RETR_EXTERNAL,
                                              cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(overlay, contours, -1, color, 2)

                # Label text
                coords = np.where(vert_mask_2d > 0)
                if len(coords[0]) > 0:
                    cy, cx = int(np.mean(coords[0])), int(np.mean(coords[1]))
                    name = VERTEBRA_NAMES[label_val]
                    cv2.putText(overlay, name, (cx - 12, cy + 6),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
                    cv2.putText(overlay, name, (cx - 12, cy + 6),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            cv2.imwrite(str(overlay_dir / f"{case_id}_overlay.png"), overlay)

    # Save results JSON
    with open(out_path / "inference_results.json", 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"Inference complete: {len(results)} cases processed")
    logger.info(f"Overlays: {overlay_dir}")
    logger.info(f"Per-vertebra masks: {mask_dir}")

    return results


def main():
    parser = argparse.ArgumentParser(description='nnU-Net Spine Inference')
    parser.add_argument('--input', type=str, required=True,
                       help='Directory with input NIfTI files')
    parser.add_argument('--output', type=str, default='results/inference',
                       help='Output directory')
    parser.add_argument('--config', type=str, default=None,
                       help='Config YAML path')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S',
    )

    if args.config is None:
        config_path = Path(__file__).parent.parent / "configs" / "training_config.yaml"
    else:
        config_path = Path(args.config)

    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    root = Path(__file__).parent.parent
    os.environ['nnUNet_raw'] = str(root / cfg['paths']['nnunet_raw'])
    os.environ['nnUNet_preprocessed'] = str(root / cfg['paths']['nnunet_preprocessed'])
    os.environ['nnUNet_results'] = str(root / cfg['paths']['nnunet_results'])

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    pred_dir = output_dir / "nnunet_raw_predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: nnU-Net prediction
    run_nnunet_predict(
        str(input_dir), str(pred_dir),
        dataset_id=cfg['nnunet']['dataset_id'],
        plans=cfg['nnunet']['plans'],
        config=cfg['nnunet']['configuration'],
        trainer=cfg['nnunet']['trainer'],
    )

    # Step 2: Post-process into per-vertebra outputs
    create_per_vertebra_outputs(str(pred_dir), str(input_dir), str(output_dir))


if __name__ == '__main__':
    main()
