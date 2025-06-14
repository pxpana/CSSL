import torch
import numpy as np
from lightly.utils.benchmarking import KNNClassifier as LightlyKNNClassifier
from typing import Any, Dict, List, Tuple, Union
from lightly.utils.benchmarking.topk import mean_topk_accuracy
from torch import Tensor

from lightly.utils.benchmarking import knn_predict

from cssl.models import BaseClassifier

class KNNClassifier(
    LightlyKNNClassifier,
    BaseClassifier
):
    def __init__(self, *args, **kwargs):
        self.metrics_logger = kwargs.pop("logger", None)
        super().__init__(*args, **kwargs)

        self.classifier_type = "KNN"

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
                feature=features,
                feature_bank=self._train_features_tensor,
                feature_labels=self._train_targets_tensor,
                num_classes=self.num_classes,
                knn_k=self.knn_k,
                knn_t=self.knn_t,
            )
            topk = mean_topk_accuracy(
                predicted_classes=predicted_classes, targets=targets, k=self.topk
            )
            log_dict = {f"val_top{k}": acc for k, acc in topk.items()}
            self.log_dict(
                log_dict, prog_bar=True, sync_dist=True, batch_size=len(targets)
            )

            self.continual_logger(predicted_classes[:, 0], targets, tasks, split="val")