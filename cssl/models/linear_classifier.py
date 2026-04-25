import torch
import numpy as np
from lightly.utils.benchmarking import LinearClassifier as LightlyLinearClassifier
from typing import Any, Dict, List, Tuple, Union
from lightly.utils.benchmarking.topk import mean_topk_accuracy
from torch import Tensor

from torch.optim import SGD, Optimizer
from lightly.utils.scheduler import CosineWarmupScheduler

from cssl.models import BaseClassifier

class LinearClassifier(
    LightlyLinearClassifier,
    BaseClassifier
):
    def __init__(self, *args, **kwargs):
        self.metrics_logger = kwargs.pop("logger", None)
        self.num_tasks = kwargs.pop("num_tasks", None)
        self.classifier_name = "Linear"
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

        if split == "val":
            self.store_val_predictions(predicted_labels, targets, tasks)

        return loss

    def training_step(
        self, 
        batch: Tuple[Tensor, ...], 
        batch_idx: int
    ) -> Tensor:
        loss = self.shared_step(batch=batch, batch_idx=batch_idx, split="train")

        return loss

    def validation_step(
        self, 
        batch: Tuple[Tensor, ...], 
        batch_idx: int
    ) -> Tensor:
        loss = self.shared_step(batch=batch, batch_idx=batch_idx, split="val")

        return loss
    
    def configure_optimizers(  # type: ignore[override]
        self,
    ) -> Tuple[List[Optimizer], List[Dict[str, Union[Any, str]]]]:
        parameters = list(self.get_trainable_parameters())

        optimizer = SGD(
            parameters,
            lr=self.get_effective_lr(),
            momentum=0.9,
            weight_decay=0.0,
        )
        scheduler = {
            "scheduler": CosineWarmupScheduler(
                optimizer=optimizer,
                warmup_epochs=int(
                    self.trainer.estimated_stepping_batches
                    / self.trainer.max_epochs
                    * 10
                ),
                max_epochs=int(self.trainer.estimated_stepping_batches),
            ),
            "interval": "step",
        }

        return [optimizer], [scheduler]
