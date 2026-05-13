#!/bin/bash
# nnU-Net Training Script for RTX 5090
# Uses ResEnc L architecture (nnU-Net Revisited, MICCAI 2024)
#
# Usage: bash scripts/train_nnunet.sh [FOLD]
# FOLD: 0-4 for individual fold, "all" for all folds (default: all)

set -e

FOLD=${1:-all}
DATASET_ID=100
TRAINER="nnUNetTrainer"
PLANNER="nnUNetPlannerResEncL"    # planner class used with -pl flag
PLANS_ID="nnUNetResEncUNetLPlans" # plans file saved by planner, used with -p flag
CONFIG="3d_fullres"

# Set nnU-Net environment variables
export nnUNet_raw="$(pwd)/data/nnUNet_raw"
export nnUNet_preprocessed="$(pwd)/data/nnUNet_preprocessed"
export nnUNet_results="$(pwd)/results/nnUNet_results"
export nnUNet_compile=0  # Disable torch.compile — RTX 5090 sm_120 not supported by cu124

echo "============================================"
echo "  nnU-Net Training — RTX 5090"
echo "============================================"
echo "  Dataset: $DATASET_ID"
echo "  Trainer: $TRAINER"
echo "  Plans: $PLANS"
echo "  Config: $CONFIG"
echo "  Fold: $FOLD"
echo "  nnUNet_raw: $nnUNet_raw"
echo "============================================"

# Step 1: Planning and preprocessing (no strict verify — Duke has direction warnings)
# Note: preprocessing is already done — skip if preprocessed dir exists
if [ ! -d "$nnUNet_preprocessed/Dataset100_SpineL1L5" ]; then
    echo ""
    echo "Step 1: Planning and preprocessing..."
    nnUNetv2_plan_and_preprocess -d $DATASET_ID -pl $PLANNER --clean
else
    echo ""
    echo "Step 1: Preprocessed data found — skipping preprocessing."
fi

# Step 3: Training
echo ""
echo "Step 3: Training..."

if [ "$FOLD" = "all" ]; then
    for f in 0 1 2 3 4; do
        echo ""
        echo "--- Training fold $f ---"
        nnUNetv2_train $DATASET_ID $CONFIG $f -tr $TRAINER -p $PLANS_ID
    done
else
    nnUNetv2_train $DATASET_ID $CONFIG $FOLD -tr $TRAINER -p $PLANS_ID
fi

# Step 4: Find best configuration
echo ""
echo "Step 4: Finding best configuration..."
nnUNetv2_find_best_configuration $DATASET_ID -c $CONFIG -tr $TRAINER -p $PLANS_ID

echo ""
echo "============================================"
echo "  Training Complete!"
echo "============================================"
echo "  Results: $nnUNet_results"
echo ""
echo "  Next: run self-training (optional)"
echo "    python scripts/self_training.py"
echo ""
echo "  Or run inference:"
echo "    python scripts/run_inference.py --input <nifti_dir>"
