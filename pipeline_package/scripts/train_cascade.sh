#!/bin/bash
# train_cascade.sh — 2-stage cascaded nnUNet training
# Stage 1: Dataset200 binary spine (3d_fullres, fold 0 only, 150 epochs → fast)
# Stage 2: Dataset201 T1/T1-FLAIR L1-L5 (3d_fullres, all 5 folds, 250 epochs)
#
# Usage:
#   bash scripts/train_cascade.sh           # train both stages
#   bash scripts/train_cascade.sh stage1    # only Stage 1
#   bash scripts/train_cascade.sh stage2    # only Stage 2
#
# Expected wall-clock time (RTX 5090):
#   Stage 1: ~1.5 hrs  (1 fold × 150 ep)
#   Stage 2: ~18-25 hrs (5 folds × 250 ep) — can run all in parallel via --c

set -e

STAGE=${1:-all}

# ─── nnUNet environment ──────────────────────────────────────────────────────
export nnUNet_raw="$(pwd)/data/nnUNet_raw"
export nnUNet_preprocessed="$(pwd)/data/nnUNet_preprocessed"
export nnUNet_results="$(pwd)/results/nnUNet_results"
export nnUNet_compile=0   # Disable torch.compile — RTX 5090 sm_120 unsupported

PLANNER="nnUNetPlannerResEncL"
PLANS="nnUNetResEncUNetLPlans"
CONFIG="3d_fullres"
# epoch-variant trainers that are built into nnUNetv2 v2.6+
TRAINER_STAGE1="nnUNetTrainer_100epochs"  # fast binary localization
TRAINER_STAGE2="nnUNetTrainer_250epochs"  # L1-L5 classification

# ─── Stage 1: Binary Spine Localization ──────────────────────────────────────
if [[ "$STAGE" == "all" || "$STAGE" == "stage1" ]]; then
    echo ""
    echo "════════════════════════════════════════════════"
    echo "  Stage 1 — Binary Spine (Dataset200)"
    echo "════════════════════════════════════════════════"

    DS_ID=200

    # Preprocess
    if [ ! -d "$nnUNet_preprocessed/Dataset200_SpineBinary" ]; then
        echo "[Stage1] Preprocessing..."
        nnUNetv2_plan_and_preprocess -d $DS_ID -pl $PLANNER --clean
    else
        echo "[Stage1] Preprocessed data found — skipping."
    fi

    # Train fold 0 only — 100 epochs is enough for binary localization
    echo "[Stage1] Training fold 0 (100 epochs via nnUNetTrainer_100epochs)..."
    nnUNetv2_train $DS_ID $CONFIG 0 -tr $TRAINER_STAGE1 -p $PLANS

    echo "[Stage1] Done! ✓"
fi

# ─── Stage 2: L1-L5 Classification ──────────────────────────────────────────
if [[ "$STAGE" == "all" || "$STAGE" == "stage2" ]]; then
    echo ""
    echo "════════════════════════════════════════════════"
    echo "  Stage 2 — L1-L5 (Dataset201)"
    echo "════════════════════════════════════════════════"

    DS_ID=201

    # Preprocess
    if [ ! -d "$nnUNet_preprocessed/Dataset201_SpineL1L5_T1" ]; then
        echo "[Stage2] Preprocessing..."
        nnUNetv2_plan_and_preprocess -d $DS_ID -pl $PLANNER --clean
    else
        echo "[Stage2] Preprocessed data found — skipping."
    fi

    # Train all 5 folds (250 epochs)
    echo "[Stage2] Training 5 folds (250 epochs each)..."
    for FOLD in 0 1 2 3 4; do
        echo ""
        echo "[Stage2] ─── Fold $FOLD ───"
        nnUNetv2_train $DS_ID $CONFIG $FOLD -tr $TRAINER_STAGE2 -p $PLANS
    done

    # Find best config
    echo ""
    echo "[Stage2] Finding best configuration..."
    nnUNetv2_find_best_configuration $DS_ID -c $CONFIG -tr $TRAINER_STAGE2 -p $PLANS

    echo "[Stage2] Done! ✓"
fi

echo ""
echo "════════════════════════════════════════════════"
echo "  Training Complete! Next step:"
echo "  python scripts/cascade_inference.py"
echo "  --input data/Sagittal_T1_FLAIR/"
echo "  --output results/cascade_final/"
echo "  --stage1-dataset 200 --stage2-dataset 201"
echo "════════════════════════════════════════════════"
