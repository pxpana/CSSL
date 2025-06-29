import math
import torch
import torch.nn.functional as F
from tqdm import tqdm
from typing import List, Dict, Tuple
from torch import Tensor
from pytorch_lightning import LightningModule
import pytorch_lightning as pl

from lightly.models.utils import get_weight_decay_parameters
from cssl.utils import LARS
from torch.optim import SGD
from lightly.utils.scheduler import CosineWarmupScheduler

from lightly.utils.benchmarking.topk import mean_topk_accuracy

from cssl.models.knn_classifier import KNNClassifier
from cssl.models.ncm_classifier import NCMClassifier

class BaseSSL(LightningModule):
    def __init__(self, backbone, config=None, loggers=None, classifier_loader=None):
        super().__init__()

        self.config = config

        self.backbone = backbone
        self.projection_head = None
        self.criterion = None
            
        self.optimizer_name = config.optimizer["name"]
        self.learning_rate = config.optimizer["train_learning_rate"]
        self.weight_decay = config.optimizer["weight_decay"]
        self.momentum = config.momentum
        self.batch_size = (config.train_batch_size*config.train_accumulate_grad_batches)/config.num_devices
        self.trust_coefficient = getattr(config, 'trust_coefficient', None)
        self.clip_lr = getattr(config, 'clip_lr', None)
        self.name = config.model_name
        self.reference_batch_size = config.reference_batch_size

        self.metrics_loggers = loggers
        self.record_classifier_every_n_epochs = config.record_classifier_every_n_epochs
        self.num_tasks = config.num_tasks
        
        self.knn_classifier = KNNClassifier(
            model=None,
            num_classes=self.config.num_classes,
            knn_k=self.config.knn_neighbours,
            knn_t=self.config.knn_temperature,
            logger=self.metrics_loggers["knn"],
            num_tasks=self.num_tasks
        )
        self.ncm_classifier = NCMClassifier(
            model=None,
            num_classes=self.config.num_classes,
            logger=self.metrics_loggers["ncm"],
            num_tasks=self.num_tasks
        )

    def setup(self, stage):
        self.knn_classifier.trainer = self.trainer
        self.ncm_classifier.trainer = self.trainer
        self.knn_classifier.log_dict = self.log_dict
        self.ncm_classifier.log_dict = self.log_dict
        
        return super().setup(stage)

    def forward(self, x):
        features = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(features)

        output = {"features": features, "projection": z}
        return output
    
    def validation_step(self, batch, batch_idx, dataloader_idx: int = 0): 
        images, targets, tasks  = batch[0], batch[1], batch[2]
        features = self.backbone(images).flatten(start_dim=1)
        features = F.normalize(features, dim=1)
        batch = (features, targets, tasks)
        
        self.knn_classifier.validation_step(batch, batch_idx, dataloader_idx=dataloader_idx)
        self.ncm_classifier.validation_step(batch, batch_idx, dataloader_idx=dataloader_idx)

    
    def on_validation_epoch_end(self):
        if not self.trainer.sanity_checking:
            self.knn_classifier.on_validation_epoch_end()
            self.ncm_classifier.on_validation_epoch_end()

        return super().on_validation_epoch_end()

    def get_effective_lr(self) -> float:
        """Compute the effective learning rate based on batch size and world size."""
        return self.learning_rate * self.batch_size * self.trainer.world_size / self.reference_batch_size

    def get_params(self):
        params, params_no_weight_decay = get_weight_decay_parameters(
            [self.backbone, self.projection_head]
        )
        return params, params_no_weight_decay
    
    def get_optimizer(self, params, params_no_weight_decay):
        # Don't use weight decay for batch norm, bias parameters, and classification
        # head to improve performance.

        if self.optimizer_name=="lars":
            optimizer = LARS([
                    {"name": f"{self.name}", "params": params},
                    {
                        "name": f"{self.name}_no_weight_decay",
                        "params": params_no_weight_decay,
                        "weight_decay": 0.0,
                    },
                ], 
                lr=self.get_effective_lr(),
                momentum=self.momentum, 
                weight_decay=self.weight_decay,
                trust_coefficient=self.trust_coefficient,
                clip_lr=self.clip_lr
            )

            scheduler = {
                "scheduler": CosineWarmupScheduler(
                    optimizer=optimizer,
                    warmup_epochs=int(self.trainer.estimated_stepping_batches / self.trainer.max_epochs * 10),
                    max_epochs=self.trainer.estimated_stepping_batches,
                ),
                "interval": "step",
            }
            
        elif self.optimizer_name=="sgd":
            optimizer = SGD(
                [
                    {"name": f"{self.name}", "params": params},
                    {
                        "name": f"{self.name}_no_weight_decay",
                        "params": params_no_weight_decay,
                        "weight_decay": 0.0,
                    },
                ],
                lr=self.get_effective_lr(),
                momentum=self.momentum,
                weight_decay=self.weight_decay,
            )

            scheduler = {
                "scheduler": CosineWarmupScheduler(
                    optimizer=optimizer,
                    warmup_epochs=0,
                    max_epochs=self.trainer.estimated_stepping_batches,
                ),
                "interval": "step",
            }

        return optimizer, scheduler
                


    
    def configure_optimizers(self):
        params, params_no_weight_decay = self.get_params()
        optimizer, scheduler = self.get_optimizer(params, params_no_weight_decay)
        return [optimizer], [scheduler]
