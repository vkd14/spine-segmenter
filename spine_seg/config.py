"""Configuration constants and dataclasses for the spine segmentation pipeline.

Mirrors the values documented in resnet_l1-l5_pipeline.pdf:
  - Six classes: 0=background, 1..5 = L1..L5
  - 5-fold ensemble, ResEncUNet Large, 3d_fullres
  - Anatomical post-processing: sagittal corridor (+/-20% LR), largest CC
    per label (>= 50 voxels), vertical SI ordering enforcement.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple


VERTEBRA_LABELS: Tuple[int, ...] = (1, 2, 3, 4, 5)

LABELS: Dict[int, str] = {
    0: "background",
    1: "L1",
    2: "L2",
    3: "L3",
    4: "L4",
    5: "L5",
}

LABEL_COLORS: Dict[int, Tuple[int, int, int]] = {
    1: (220, 38, 38),    # L1 red
    2: (34, 197, 94),    # L2 green
    3: (59, 130, 246),   # L3 blue
    4: (249, 115, 22),   # L4 orange
    5: (168, 85, 247),   # L5 purple
}


@dataclass
class ModelConfig:
    """Mirrors training-time CLI flags exactly.

    From the PDF (Section 5.3):
        nnUNetv2_predict -d 100 -c 3d_fullres
            -tr nnUNetTrainer -p nnUNetResEncUNetLPlans
            -f 0 1 2 3 4 -chk checkpoint_best.pth -step_size 0.5
    """

    dataset_name: str = "Dataset100_SpineL1L5"
    dataset_id: int = 100
    trainer: str = "nnUNetTrainer"
    plans: str = "nnUNetResEncUNetLPlans"
    configuration: str = "3d_fullres"
    checkpoint_name: str = "checkpoint_best.pth"
    folds: Tuple[int, ...] = (0, 1, 2, 3, 4)
    tile_step_size: float = 0.5
    use_mirroring: bool = True
    use_gaussian: bool = True


@dataclass
class PostProcessConfig:
    """3-step anatomical post-processing parameters from the PDF.

    Section 5.4:
        Step 1: sagittal strip corridor +/- sagittal_strip_fraction * N_LR
        Step 2: largest connected component per label, drop if < min_component_voxels
        Step 3: vertical SI ordering enforcement (drop labels violating L1->L5 monotonicity)
    """

    sagittal_strip_fraction: float = 0.20
    min_component_voxels: int = 50
    cc_connectivity: int = 26
