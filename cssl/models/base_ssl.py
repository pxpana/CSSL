import math
from pytorch_lightning import LightningModule

from lightly.models.utils import get_weight_decay_parameters
from cssl.utils import LARS
from torch.optim import SGD
from lightly.utils.scheduler import CosineWarmupScheduler

class BaseSSL(LightningModule):
    def __init__(self, backbone, config=None):
        super().__init__()

        self.backbone = backbone
        self.projection_head = None
        self.criterion = None

        if config != None:
            self.optimizer_name = config.optimizer["name"]
            self.learning_rate = config.optimizer["train_learning_rate"]
            self.weight_decay = config.optimizer["weight_decay"]
            self.momentum = config.momentum
            self.batch_size = (config.train_batch_size*config.train_accumulate_grad_batches)/config.num_devices
            self.trust_coefficient = getattr(config, 'trust_coefficient', None)
            self.clip_lr = getattr(config, 'clip_lr', None)
            self.name = config.model_name
            self.reference_batch_size = config.reference_batch_size

    def forward(self, x):
        features = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(features)

        output = {"features": features, "projection": z}
        return output
    
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
