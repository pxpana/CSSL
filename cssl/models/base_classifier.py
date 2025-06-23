import math
import numpy as np
from typing import Any, Dict, List, Tuple, Union
from pytorch_lightning import LightningModule

from lightly.models.utils import get_weight_decay_parameters
from cssl.utils import LARS
from torch.optim import SGD, Optimizer
from lightly.utils.scheduler import CosineWarmupScheduler
from torch.optim.lr_scheduler import MultiStepLR

class BaseClassifier(LightningModule):
    def __init__(self):
        super().__init__()

        self.predicted_labels = []
        self.targets = []
        self.tasks = []

        self.lr_decay_steps = [60, 80]


    def on_validation_epoch_end(self):
        if not self.trainer.sanity_checking:
            self.continual_logger(
                predicted_labels=np.array(self.predicted_labels), 
                targets=np.array(self.targets), 
                tasks=np.array(self.tasks), 
                split="val"
            )
            log_dict = {
                f"Accuracy": self.metrics_logger.accuracy,
                f"Ave. Accuracy": self.metrics_logger.average_accuracy,
                # f"Accuracy 1:t": np.mean(np.array(self.metrics_logger.accuracy_per_task)[:(self.metrics_logger.current_task+1)]),
                # f"Accuracy 1:T": np.mean(np.array(self.metrics_logger.accuracy_per_task)),
                # f"AIC": self.metrics_logger.average_incremental_accuracy,
                # f"BWT": self.metrics_logger.backward_transfer,
                f"FWT": self.metrics_logger.forward_transfer,
                # f"PBWT": self.metrics_logger.positive_backward_transfer,
                # f"Remembering": self.metrics_logger.remembering,
                f"Forgetting": self.metrics_logger.forgetting,
            }
            self.metrics_logger.end_epoch()
            
            predicted_labels = np.array(self.predicted_labels)
            targets = np.array(self.targets)
            tasks = np.array(self.tasks)

            for task_idx in range(5):
                mask = task_idx==tasks
                task_predicted_labels = predicted_labels[mask]
                task_targets = targets[mask]

                log_dict[f"val_acc1_task{task_idx}"] = np.mean(task_predicted_labels == task_targets)

            self.log_dict(log_dict, sync_dist=True, prog_bar=True)
        self.predicted_labels = []
        self.targets = []
        self.tasks = []

        return super().on_validation_epoch_end()

    def teardown(self, stage):
        self.metrics_logger.end_task()

        return super().teardown(stage)


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

    