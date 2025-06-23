import math
from typing import List, Dict, Tuple
from torch import Tensor
from pytorch_lightning import LightningModule

from lightly.models.utils import get_weight_decay_parameters
from cssl.utils import LARS
from torch.optim import SGD
from lightly.utils.scheduler import CosineWarmupScheduler
from lightly.utils.benchmarking import OnlineLinearClassifier

from lightly.utils.benchmarking.topk import mean_topk_accuracy

class BaseSSL(LightningModule):
    def __init__(self, backbone, config=None):
        super().__init__()

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

        self.online_classifier = OnlineLinearClassifier(feature_dim=config.feature_dim, num_classes=config.num_classes)
        self.classifier_learning_rate = config.optimizer["classifier_learning_rate"]

    def forward(self, x):
        features = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(features)

        output = {"features": features, "projection": z}
        return output

    def validation_step(
        self, batch: Tuple[Tensor, Tensor, List[str]], batch_idx: int
    ) -> Tensor:
        images, targets = batch[0], batch[1]
        features = self.backbone(images).flatten(start_dim=1)
        cls_loss, cls_log = self.online_classifier.validation_step(
            (features.detach(), targets), batch_idx
        )

        self.log_dict(cls_log, prog_bar=True, sync_dist=True, batch_size=len(targets))
        return cls_loss

    def online_training_step(self, batch: Tuple[Tensor, Tensor, List[str]], batch_idx: int) -> Tensor:
        images, targets = batch[0], batch[1]
        features = self.backbone(images).flatten(start_dim=1)
        cls_loss, cls_log = self.online_classifier.training_step(
            (features.detach(), targets), batch_idx
        )

        self.log_dict(cls_log, prog_bar=True, sync_dist=True, batch_size=len(targets))
        return cls_loss

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
                    {
                        "name": f"{self.name}_online_classifier",
                        "params": self.online_classifier.parameters(),
                        "lr": self.classifier_learning_rate,
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
