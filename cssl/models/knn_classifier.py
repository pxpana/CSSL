import torch
import numpy as np
from lightly.utils.benchmarking import KNNClassifier as LightlyKNNClassifier
from typing import Any, Dict, List, Tuple, Union
from lightly.utils.benchmarking.topk import mean_topk_accuracy
from torch import Tensor

from lightly.utils.benchmarking import knn_predict

from cssl.models.base_classifier import BaseClassifier

class KNNClassifier(
    LightlyKNNClassifier,
    BaseClassifier
):
    def __init__(self, *args, **kwargs):
        self.metrics_logger = kwargs.pop("logger", None)
        self.num_tasks = kwargs.pop("num_tasks", None)
        super().__init__(*args, **kwargs)

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

            assert self._train_features_tensor is not None
            assert self._train_targets_tensor is not None

            predicted_classes = knn_predict(
                feature=features.cpu(),
                feature_bank=self._train_features_tensor,
                feature_labels=self._train_targets_tensor,
                num_classes=self.num_classes,
                knn_k=self.knn_k,
                knn_t=self.knn_t,
            )

            self.store_val_predictions(predicted_classes[:, 0], targets, tasks)

    def concat_train_features(self) -> None:
        if self._train_features and self._train_targets:
            # Features and targets have size (world_size, batch_size, dim) and
            # (world_size, batch_size) after gather. For non-distributed training,
            # features and targets have size (batch_size, dim) and (batch_size,).

            features = torch.cat(self._train_features, dim=0)
            self._train_features = []
            targets = torch.cat(self._train_targets, dim=0)
            self._train_targets = []
            # Reshape to (dim, world_size * batch_size)
            features = features.flatten(end_dim=-2).t().contiguous()
            self._train_features_tensor = features
            # Reshape to (world_size * batch_size,)
            targets = targets.flatten().t().contiguous()
            self._train_targets_tensor = targets


