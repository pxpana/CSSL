import numpy as np
from lightly.utils.benchmarking import LinearClassifier
from typing import Any, Dict, List, Tuple, Union
from lightly.utils.benchmarking.topk import mean_topk_accuracy
from torch import Tensor

class Classifier(LinearClassifier):
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

        return loss

    def validation_step(
        self, 
        batch: Tuple[Tensor, ...], 
        batch_idx: int
    ) -> Tensor:
        loss = self.shared_step(batch=batch, batch_idx=batch_idx, split="val")

        return loss
    
    def on_validation_epoch_end(self):
        if not self.trainer.sanity_checking:
            accuracy_per_task = self.metrics_logger.accuracy_per_task
            log_dict = {
                f"Task Accuracy": accuracy_per_task[-1],
                f"Accuracy": np.mean(accuracy_per_task),
                f"AIC": self.metrics_logger.average_incremental_accuracy,
                f"BWT": self.metrics_logger.backward_transfer,
                f"FWT": self.metrics_logger.forward_transfer,
                f"PBWT": self.metrics_logger.positive_backward_transfer,
                f"Remembering": self.metrics_logger.remembering,
                f"Forgetting": self.metrics_logger.forgetting,
            }

            self.log_dict(log_dict, sync_dist=True, prog_bar=True)
            self.metrics_logger.end_epoch()

        return super().on_validation_epoch_end()
    
    def on_train_end(self):
        self.metrics_logger.end_task()

        return super().on_train_end()


    def continual_logger(self, predicted_labels, targets, tasks, split):

        if split=="val":
            self.metrics_logger.add(
                [
                    predicted_labels, 
                    targets, 
                    tasks
                ], 
                subset="test" if split=="val" else "train"
            )