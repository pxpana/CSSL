import math
import numpy as np
from pytorch_lightning import LightningModule

from lightly.models.utils import get_weight_decay_parameters
from cssl.utils import LARS
from torch.optim import SGD
from lightly.utils.scheduler import CosineWarmupScheduler

class BaseClassifier(LightningModule):
    def __init__(self):
        super().__init__()

    def on_validation_epoch_end(self):
        if not self.trainer.sanity_checking:
            log_dict = {
                f"{self.classifier_type}_Accuracy": np.mean(np.array(self.metrics_logger.accuracy_per_task)[:(self.metrics_logger.current_task+1)]),
                f"{self.classifier_type}_Accuracy 1:T": np.mean(np.array(self.metrics_logger.accuracy_per_task)),
                f"{self.classifier_type}_AIC": self.metrics_logger.average_incremental_accuracy,
                f"{self.classifier_type}_BWT": self.metrics_logger.backward_transfer,
                f"{self.classifier_type}_FWT": self.metrics_logger.forward_transfer,
                f"{self.classifier_type}_PBWT": self.metrics_logger.positive_backward_transfer,
                f"{self.classifier_type}_Remembering": self.metrics_logger.remembering,
                f"{self.classifier_type}_Forgetting": self.metrics_logger.forgetting,
            }
            self.metrics_logger.end_epoch()
            self.log_dict(log_dict, sync_dist=True, prog_bar=True)

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

    