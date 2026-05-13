"""
Self-Training with FLARE22 Recipe.

After initial nnU-Net training on SPIDER:
1. Predict all 5 folds on unlabeled T1_FLAIR data
2. Keep voxels where >= 4/5 folds agree (high-confidence pseudo-labels)
3. Add pseudo-labeled data to training set
4. Retrain nnU-Net
"""

import os
import logging
import subprocess
import numpy as np
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

try:
    import nibabel as nib
except ImportError:
    raise ImportError("nibabel required")

try:
    import yaml
except ImportError:
    raise ImportError("PyYAML required")


def predict_all_folds(
    dataset_id: int,
    input_dir: str,
    output_base: str,
    plans: str = "nnUNetResEncUNetLPlans",
    config: str = "3d_fullres",
    trainer: str = "nnUNetTrainer",
):
    """Run nnU-Net inference with all 5 folds."""
    output_base = Path(output_base)

    for fold in range(5):
        fold_output = output_base / f"fold_{fold}"
        fold_output.mkdir(parents=True, exist_ok=True)

        cmd = [
            "nnUNetv2_predict",
            "-i", str(input_dir),
            "-o", str(fold_output),
            "-d", str(dataset_id),
            "-c", config,
            "-tr", trainer,
            "-p", plans,
            "-f", str(fold),
            "--disable_tta",  # faster, minimal quality loss
        ]

        logger.info(f"Predicting fold {fold}...")
        subprocess.run(cmd, check=True)

    logger.info("All 5 folds predicted")


def generate_consensus_pseudolabels(
    fold_predictions_dir: str,
    output_dir: str,
    min_agreement: int = 4,
    n_folds: int = 5,
) -> int:
    """Generate pseudo-labels from fold predictions using majority voting.

    Only keeps voxels where >= min_agreement folds agree on the label.
    """
    pred_base = Path(fold_predictions_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Get all prediction files from fold 0
    fold0_dir = pred_base / "fold_0"
    pred_files = sorted(fold0_dir.glob("*.nii.gz"))

    count = 0

    for pf in pred_files:
        fname = pf.name

        # Load all fold predictions
        all_preds = []
        for fold in range(n_folds):
            fold_file = pred_base / f"fold_{fold}" / fname
            if not fold_file.exists():
                break
            pred = nib.load(str(fold_file)).get_fdata().astype(np.int32)
            all_preds.append(pred)

        if len(all_preds) < n_folds:
            continue

        # Stack and compute agreement
        stacked = np.stack(all_preds, axis=0)  # (5, D, H, W)

        # For each voxel, find the most common label and its count
        from scipy.stats import mode
        mode_result = mode(stacked, axis=0, keepdims=False)
        consensus_label = mode_result.mode.astype(np.uint8)
        agreement_count = mode_result.count

        # Zero out voxels with insufficient agreement
        consensus_label[agreement_count < min_agreement] = 0

        # Check if any L1-L5 labels remain
        if consensus_label.max() == 0:
            continue

        # Save
        ref_img = nib.load(str(pf))
        pseudo_nii = nib.Nifti1Image(consensus_label, ref_img.affine, ref_img.header)
        nib.save(pseudo_nii, str(out_path / fname))
        count += 1

    logger.info(f"Generated {count} consensus pseudo-labels "
               f"(>= {min_agreement}/{n_folds} agreement)")
    return count


def add_pseudolabels_to_dataset(
    pseudo_labels_dir: str,
    pseudo_images_dir: str,
    nnunet_dataset_dir: str,
    start_idx: int = 1000,
) -> int:
    """Add pseudo-labeled data to nnU-Net dataset for retraining."""
    pseudo_dir = Path(pseudo_labels_dir)
    images_dir = Path(pseudo_images_dir)
    dataset_dir = Path(nnunet_dataset_dir)

    out_images = dataset_dir / "imagesTr"
    out_labels = dataset_dir / "labelsTr"

    pseudo_files = sorted(pseudo_dir.glob("*.nii.gz"))
    count = 0

    for pf in pseudo_files:
        # Find corresponding image
        img_name = pf.name
        img_file = images_dir / img_name

        if not img_file.exists():
            # Try with _0000 suffix
            stem = pf.stem.replace('.nii', '')
            img_file = images_dir / f"{stem}.nii.gz"
            if not img_file.exists():
                continue

        case_id = f"pseudo_{start_idx + count:04d}"

        # Copy image (add _0000 suffix if needed)
        import shutil
        shutil.copy2(str(img_file), str(out_images / f"{case_id}_0000.nii.gz"))
        shutil.copy2(str(pf), str(out_labels / f"{case_id}.nii.gz"))

        count += 1

    logger.info(f"Added {count} pseudo-labeled cases to dataset")
    return count


def run_self_training(config_path: str = None):
    """Full self-training pipeline."""
    if config_path is None:
        config_path = Path(__file__).parent.parent / "configs" / "training_config.yaml"

    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    root = Path(__file__).parent.parent
    st_cfg = cfg['self_training']

    if not st_cfg.get('enabled', False):
        logger.info("Self-training disabled in config")
        return

    dataset_id = cfg['nnunet']['dataset_id']
    plans = cfg['nnunet']['plans']
    config = cfg['nnunet']['configuration']
    trainer = cfg['nnunet']['trainer']

    # Set nnU-Net env
    os.environ['nnUNet_raw'] = str(root / cfg['paths']['nnunet_raw'])
    os.environ['nnUNet_preprocessed'] = str(root / cfg['paths']['nnunet_preprocessed'])
    os.environ['nnUNet_results'] = str(root / cfg['paths']['nnunet_results'])

    n_rounds = st_cfg.get('rounds', 3)
    min_agreement = st_cfg.get('min_fold_agreement', 4)

    # Unlabeled data directory
    t1_flair_dir = root / cfg['paths']['t1_flair_nifti']

    for round_idx in range(n_rounds):
        logger.info(f"\n{'='*60}")
        logger.info(f"  Self-Training Round {round_idx + 1}/{n_rounds}")
        logger.info(f"{'='*60}\n")

        # Step 1: Predict all folds on unlabeled data
        pred_dir = root / "results" / "self_training" / f"round{round_idx}" / "predictions"
        predict_all_folds(
            dataset_id, str(t1_flair_dir), str(pred_dir),
            plans=plans, config=config, trainer=trainer,
        )

        # Step 2: Generate consensus pseudo-labels
        pseudo_dir = root / "results" / "self_training" / f"round{round_idx}" / "pseudo_labels"
        n_pseudo = generate_consensus_pseudolabels(
            str(pred_dir), str(pseudo_dir),
            min_agreement=min_agreement,
        )

        if n_pseudo == 0:
            logger.warning("No pseudo-labels generated. Stopping self-training.")
            break

        # Step 3: Add to dataset and retrain
        dataset_dir = root / cfg['paths']['nnunet_raw'] / cfg['nnunet']['dataset_name']
        add_pseudolabels_to_dataset(
            str(pseudo_dir), str(t1_flair_dir),
            str(dataset_dir), start_idx=1000 + round_idx * 1000,
        )

        # Step 4: Re-preprocess and retrain
        logger.info("Re-preprocessing dataset...")
        subprocess.run([
            "nnUNetv2_plan_and_preprocess",
            "-d", str(dataset_id),
            "-pl", plans,
            "--clean",
        ], check=True)

        logger.info("Retraining all folds...")
        for fold in range(5):
            subprocess.run([
                "nnUNetv2_train",
                str(dataset_id), config, str(fold),
                "-tr", trainer,
                "-p", plans,
            ], check=True)

        logger.info(f"Round {round_idx + 1} complete: +{n_pseudo} pseudo-labels")

    logger.info("\nSelf-training complete!")


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S',
    )
    run_self_training()
