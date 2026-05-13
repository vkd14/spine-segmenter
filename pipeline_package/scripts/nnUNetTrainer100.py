"""
Custom nnUNet trainer for short fine-tuning runs (100 epochs).
Place this file so nnUNet can discover it via its plugin registry.
"""
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainer100(nnUNetTrainer):
    """nnUNetTrainer capped at 100 epochs — for fast multi-fold fine-tuning."""
    def __init__(self, plans, configuration, fold, dataset_json,
                 unpack_dataset=True, device='cuda'):
        super().__init__(plans, configuration, fold, dataset_json,
                         unpack_dataset, device)
        self.num_epochs = 100
