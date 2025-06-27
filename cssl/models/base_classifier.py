import math
import numpy as np
from typing import Any, Dict, List, Tuple, Union
from pytorch_lightning import LightningModule

from lightly.models.utils import get_weight_decay_parameters
from cssl.utils import LARS
from torch.optim import SGD, Optimizer
from lightly.utils.scheduler import CosineWarmupScheduler
from torch.optim.lr_scheduler import MultiStepLR

from torch import Tensor

class BaseClassifier(LightningModule):
    def __init__(self):
        super().__init__()

        self.predicted_labels = []
        self.targets = []
        self.tasks = []

        self.lr_decay_steps = [60, 80]


    def on_validation_epoch_end(self):
        if not self.trainer.sanity_checking:
            log_dict = {}

            predicted_labels = np.array(self.predicted_labels)
            targets = np.array(self.targets)
            tasks = np.array(self.tasks)

            for task_idx in range(self.num_tasks):
                mask = task_idx==tasks
                task_predicted_labels = predicted_labels[mask]
                task_targets = targets[mask]

                log_dict[f"{self.classifier_name} val_acc1_task{task_idx+1}"] = np.mean(task_predicted_labels == task_targets)

            if self.metrics_logger is not None:
                self.continual_logger(
                    predicted_labels=predicted_labels, 
                    targets=targets, 
                    tasks=tasks, 
                    split="val"
                )
                log_dict[f"{self.classifier_name} Accuracy"] = self.metrics_logger.accuracy
                log_dict[f"{self.classifier_name} Ave. Accuracy"] = self.metrics_logger.average_accuracy
                log_dict[f"{self.classifier_name} AIC"] = self.metrics_logger.average_incremental_accuracy
                log_dict[f"{self.classifier_name} BWT"] = self.metrics_logger.backward_transfer
                log_dict[f"{self.classifier_name} FWT"] = self.metrics_logger.forward_transfer
                log_dict[f"{self.classifier_name} PBWT"] = self.metrics_logger.positive_backward_transfer
                log_dict[f"{self.classifier_name} Remembering"] = self.metrics_logger.remembering
                log_dict[f"{self.classifier_name} Forgetting"] = self.metrics_logger.forgetting
                
                self.metrics_logger.end_epoch()

            self.log_dict(log_dict, sync_dist=True, prog_bar=True)

        self.predicted_labels = []
        self.targets = []
        self.tasks = []

        return super().on_validation_epoch_end()

    def teardown(self, stage):
        if self.metrics_logger is not None:
            self.metrics_logger.end_task()

        return super().teardown(stage)

    def store_val_predictions(self, predicted_labels, targets, tasks):
        self.predicted_labels.extend(predicted_labels.detach().cpu().tolist())
        self.targets.extend(targets.detach().cpu().tolist())
        self.tasks.extend(tasks.detach().cpu().tolist())

    def continual_logger(self, predicted_labels, targets, tasks, split):
        if split=="val" and self.metrics_logger is not None:
            self.metrics_logger.add(
                [
                    predicted_labels, 
                    targets, 
                    tasks
                ], 
                subset="test" if split=="val" else "train"
            )

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
            "scheduler": MultiStepLR(
                optimizer=optimizer, 
                milestones=self.lr_decay_steps, 
                gamma=0.1
            ),
            "interval": "step",
        }

        return [optimizer], [scheduler]

    def append_train_features(self, features: Tensor, targets: Tensor) -> None:
        self._train_features.append(features)
        self._train_targets.append(targets)

    