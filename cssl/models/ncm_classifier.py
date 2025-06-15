import torch
import numpy as np
from lightly.utils.benchmarking import KNNClassifier as LightlyKNNClassifier
from typing import Any, Dict, List, Tuple, Union
from lightly.utils.benchmarking.topk import mean_topk_accuracy
from torch import Tensor

from cssl.models import BaseClassifier

class NCMClassifier(
    LightlyKNNClassifier,
    BaseClassifier
):
    def __init__(self, *args, **kwargs):
        self.metrics_logger = kwargs.pop("logger", None)
        super().__init__(*args, **kwargs)
        
        self._class_means = None

    def concat_train_features(self) -> None:
        super().concat_train_features()

        self._train_features_tensor = self._train_features_tensor.T

        # Compute class means
        self._class_means = []
        for class_idx in range(self.num_classes):
            mask = self._train_targets_tensor == class_idx
            if mask.any():
                class_features = self._train_features_tensor[mask]
                class_mean = class_features.mean(dim=0, keepdim=False)
                self._class_means.append(class_mean)
            else:
                # Handle case where a class has no samples
                self._class_means.append(torch.zeros(1, self._train_features_tensor.size(1)))
        
        self._class_means = torch.stack(self._class_means)
        self._class_means = self._class_means.to(self._train_features_tensor.device)

    def ncm_predict(self, features: torch.Tensor) -> torch.Tensor:
        if self._class_means is None:
            raise ValueError("Class means not computed. Call concat_train_features first.")
            
        # Compute distances to class means
        distances = torch.cdist(features, self._class_means, p=2)  # [batch_size, num_classes]
        # Convert distances to similarities (higher is better)
        similarities = -distances
            
        return similarities

    def validation_step(self, batch, batch_idx: int, dataloader_idx: int) -> None:
        images, targets, tasks = batch[0], batch[1], batch[2]
        features = self(images)

        if dataloader_idx == 0:
            # The first dataloader is the training dataloader.
            self.append_train_features(features=features, targets=targets)
        else:
            if batch_idx == 0 and dataloader_idx == 1:
                # Concatenate train features when starting the validation dataloader.
                self.concat_train_features()

            assert self._class_means is not None
            predicted_scores = self.ncm_predict(features=features)
            _, predicted_classes = predicted_scores.topk(max(self.topk))

            self.continual_logger(predicted_classes[:, 0], targets, tasks, split="val")