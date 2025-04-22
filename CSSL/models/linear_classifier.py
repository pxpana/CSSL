from lightly.utils.benchmarking import LinearClassifier
from typing import Any, Dict, List, Tuple, Union
from lightly.utils.benchmarking.topk import mean_topk_accuracy
from torch import Tensor

class Classifier(LinearClassifier):
    def __init__(self, *args, **kwargs):
        self.metrics_logger = kwargs.pop("logger", None)
        self.current_task = kwargs.pop("current_task", None)
        super().__init__(*args, **kwargs)

    def shared_step(
        self, 
        batch: Tuple[Tensor, ...], 
        batch_idx: int,
        split: str
    ) -> Tuple[Tensor, Dict[int, Tensor]]:
        images, targets = batch[0], batch[1]
        predictions = self.forward(images)
        loss = self.criterion(predictions, targets)
        _, predicted_labels = predictions.topk(1)
        predicted_labels = predicted_labels.flatten()

        batch_size = len(batch[1])
        self.log(
            f"{split}_loss", loss, prog_bar=True, sync_dist=True, batch_size=batch_size
        )
        log_dict = self.continual_logger(predicted_labels, targets, split)
        self.log_dict(log_dict, sync_dist=True, batch_size=batch_size, prog_bar=True)

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

    def continual_logger(self, predicted_labels, targets, split):
        self.metrics_logger.add([predicted_labels, targets, [self.current_task]], subset="test" if split=="val" else "train")
        log_dict = {
            f"Accuracy_{split}": self.metrics_logger.accuracy,
            f"AIC_{split}": self.metrics_logger.average_incremental_accuracy,
            f"BWT_{split}": self.metrics_logger.backward_transfer,
            f"FWT_{split}": self.metrics_logger.forward_transfer,
            f"PBWT_{split}": self.metrics_logger.positive_backward_transfer,
            f"Remembering_{split}": self.metrics_logger.remembering,
            f"Forgetting_{split}": self.metrics_logger.forgetting,
        }

        return log_dict