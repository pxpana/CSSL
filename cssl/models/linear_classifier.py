import torch
import numpy as np
from lightly.utils.benchmarking import LinearClassifier as LightlyLinearClassifier
from typing import Any, Dict, List, Tuple, Union
from lightly.utils.benchmarking.topk import mean_topk_accuracy
from torch import Tensor

from cssl.models import BaseClassifier

class LinearClassifier(
    LightlyLinearClassifier,
    BaseClassifier
):
    def __init__(self, *args, **kwargs):
        self.metrics_logger = kwargs.pop("logger", None)
        super().__init__(*args, **kwargs)

    def shared_step(
        self, 
        batch: Tuple[Tensor, ...], 
        batch_idx: int,
        split: str
    ) -> Tuple[Tensor, Dict[int, Tensor]]:
        images, targets, tasks = batch[0], batch[1], batch[2]

        predictions = self.forward(images)
        loss = self.criterion(predictions, targets)
        _, predicted_labels = predictions.topk(1)
        predicted_labels = predicted_labels.flatten()

        batch_size = len(batch[1])
        self.log(
            f"{split}_loss", loss, prog_bar=True, sync_dist=True, batch_size=batch_size
        )
        self.continual_logger(predicted_labels, targets, tasks, split)

        return loss

    def training_step(
        self, 
        batch: Tuple[Tensor, ...], 
        batch_idx: int
    ) -> Tensor:
        loss = self.shared_step(batch=batch, batch_idx=batch_idx, split="train")
        images, targets, tasks = batch[0], batch[1], batch[2]

        return loss

    def validation_step(
        self, 
        batch: Tuple[Tensor, ...], 
        batch_idx: int
    ) -> Tensor:
        loss = self.shared_step(batch=batch, batch_idx=batch_idx, split="val")

        return loss