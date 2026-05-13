#!/bin/bash
# Train remaining folds 1-4 with 100 epochs each.
# Uses fold_0 checkpoint_best.pth as pretrained weights.
# Uses nnUNetTrainer100 (custom trainer capped at 100 epochs).
#
# Usage:
#   bash scripts/train_folds_100ep.sh          # runs folds 1 2 3 4
#   bash scripts/train_folds_100ep.sh 2        # single fold
#   bash scripts/train_folds_100ep.sh "2 3"    # specific folds

set -e

DATASET_ID=100
TRAINER="nnUNetTrainer100"      # Custom trainer: 100 epoch cap
PLANS_ID="nnUNetResEncUNetLPlans"
CONFIG="3d_fullres"

export nnUNet_raw="$(pwd)/data/nnUNet_raw"
export nnUNet_preprocessed="$(pwd)/data/nnUNet_preprocessed"
export nnUNet_results="$(pwd)/results/nnUNet_results"
export nnUNet_compile=0

BASE="$(pwd)/results/nnUNet_results/Dataset100_SpineL1L5/nnUNetTrainer__${PLANS_ID}__${CONFIG}"
BEST_CKPT="${BASE}/fold_0/checkpoint_best.pth"

echo "============================================"
echo "  nnU-Net — Folds 1-4, 100 Epochs Each"
echo "============================================"
echo "  Trainer:   $TRAINER (100-epoch cap)"
echo "  Plans:     $PLANS_ID"
echo "  Pretrained: fold_0/checkpoint_best.pth"
echo "============================================"

if [ ! -f "$BEST_CKPT" ]; then
    echo "ERROR: checkpoint_best.pth not found at:"
    echo "  $BEST_CKPT"
    exit 1
fi

FOLDS="${1:-1 2 3 4}"

for f in $FOLDS; do
    echo ""
    echo "--- Training fold $f (100 epochs, warm-started from fold_0 best) ---"
    PATH="$(pwd)/.venv/bin:$PATH" nnUNetv2_train $DATASET_ID $CONFIG $f \
        -tr $TRAINER \
        -p $PLANS_ID \
        -pretrained_weights "$BEST_CKPT"
done

echo ""
echo "--- Finding best configuration across all completed folds ---"
# Compare fold_0 (full 1000ep trainer) with folds 1-4 (100ep trainer)
PATH="$(pwd)/.venv/bin:$PATH" nnUNetv2_find_best_configuration \
    $DATASET_ID -c $CONFIG -tr $TRAINER -p $PLANS_ID || true

echo ""
echo "============================================"
echo "  Folds complete! Run inference with:"
echo "    python scripts/run_inference.py --input <nifti_dir>"
echo "============================================"
