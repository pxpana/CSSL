import math
import torch
import torch.nn.functional as F
from tqdm import tqdm
from typing import List, Dict, Tuple
from torch import Tensor
from pytorch_lightning import LightningModule
import pytorch_lightning as pl

from cssl.utils import LARS
#from lightly.utils.lars import LARS
from torch.optim import SGD
from lightly.utils.scheduler import CosineWarmupScheduler

from lightly.utils.benchmarking.topk import mean_topk_accuracy

from cssl.models.online_linear_classifier import OnlineLinearClassifier
from cssl.models.knn_classifier import KNNClassifier
from cssl.models.ncm_classifier import NCMClassifier

class BaseSSL(LightningModule):
    def __init__(self, backbone, config=None, loggers=None):
        super().__init__()

        self.config = config

        self.backbone = backbone
        self.projection_head = None
        self.criterion = None

        self.metrics_loggers = loggers
        self.num_tasks = config.num_tasks

        if self.config.use_online_classifier:
            self.online_classifier = OnlineLinearClassifier(
                num_classes=self.config.num_classes, 
                feature_dim=self.config.feature_dim,
                logger=self.metrics_loggers["online_linear"],
                num_tasks=self.num_tasks
            )
        
        if self.config.use_knn_classifier:
            self.knn_classifier = KNNClassifier(
                model=None,
                num_classes=self.config.num_classes,
                knn_k=self.config.knn_neighbours,
                knn_t=self.config.knn_temperature,
                logger=self.metrics_loggers["knn"],
                num_tasks=self.num_tasks
            )

        if self.config.use_ncm_classifier:
            self.ncm_classifier = NCMClassifier(
                model=None,
                num_classes=self.config.num_classes,
                logger=self.metrics_loggers["ncm"],
                num_tasks=self.num_tasks
            )

    def setup(self, stage):
        if self.config.use_knn_classifier:
            self.knn_classifier.trainer = self.trainer
            self.knn_classifier.log_dict = self.log_dict
        if self.config.use_ncm_classifier:
            self.ncm_classifier.trainer = self.trainer
            self.ncm_classifier.log_dict = self.log_dict
        if self.config.use_online_classifier:
            self.online_classifier.trainer = self.trainer
            self.online_classifier.log_dict = self.log_dict
        
        return super().setup(stage)
    
    def validation_step(self, batch, batch_idx, dataloader_idx: int = 0): 
        images, targets, tasks  = batch[0], batch[1], batch[2]
        features = self.backbone(images).flatten(start_dim=1)

        if self.config.use_online_classifier:
            if dataloader_idx == 1:  # only run on the validation dataloader
                online_log = self.online_classifier.validation_step((features.detach(), targets, tasks), batch_idx)
                self.log_dict(online_log, prog_bar=True, sync_dist=True, batch_size=len(targets))

        features = F.normalize(features, dim=1)
        batch = (features, targets, tasks)

        if self.config.use_knn_classifier:
            self.knn_classifier.validation_step(batch, batch_idx, dataloader_idx=dataloader_idx)
        if self.config.use_ncm_classifier:
            self.ncm_classifier.validation_step(batch, batch_idx, dataloader_idx=dataloader_idx)

    
    def on_validation_epoch_end(self):
        if not self.trainer.sanity_checking:
            if self.config.use_online_classifier:
                log_dict = self.online_classifier.on_validation_epoch_end()
                self.log_dict(log_dict, sync_dist=True, prog_bar=True)
            if self.config.use_knn_classifier:
                log_dict = self.knn_classifier.on_validation_epoch_end()
                self.log_dict(log_dict, sync_dist=True, prog_bar=True)
            if self.config.use_ncm_classifier:
                log_dict = self.ncm_classifier.on_validation_epoch_end()
                self.log_dict(log_dict, sync_dist=True, prog_bar=True)

        return super().on_validation_epoch_end()
    
    def teardown(self, stage):
        if self.config.use_online_classifier:
            self.online_classifier.teardown(stage)
        if self.config.use_knn_classifier:
            self.knn_classifier.teardown(stage)
        if self.config.use_ncm_classifier:
            self.ncm_classifier.teardown(stage)
        return super().teardown(stage)
    
    def configure_optimizers(self):
        # Don't use weight decay for batch norm, bias parameters, and classification
        # head to improve performance.

        params, params_no_weight_decay = self.get_params()

        if self.config.use_online_classifier:
            online_params = self.online_classifier.parameters()
        else:
            online_params = []

        if self.config.optimizer["name"]=="lars":
            optimizer = LARS([
                    {"name": f"{self.config.model}", "params": params},
                    {
                        "name": f"{self.config.model}_no_weight_decay",
                        "params": params_no_weight_decay,
                        "weight_decay": 0.0,
                    },
                    {
                        "name": "online_classifier",
                        "params": online_params,
                        "weight_decay": 0.0,
                    },
                ], 
                lr=self.get_effective_lr(),
                momentum=self.config.optimizer["momentum"], 
                weight_decay=self.config.optimizer["weight_decay"],
                trust_coefficient=self.config.optimizer["trust_coefficient"],
                clip_lr=self.config.optimizer["clip_lr"]
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
                    {
                        "name": "online_classifier",
                        "params": self.online_classifier.parameters(),
                        "weight_decay": 0.0,
                    },
                ],
                lr=self.get_effective_lr(),
                momentum=self.config.optimizer_momentum,
                weight_decay=self.config.weight_decay,
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
                
