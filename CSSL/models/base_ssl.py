import math
from pytorch_lightning import LightningModule

from lightly.models.utils import get_weight_decay_parameters
from utils import LARS
from lightly.utils.scheduler import CosineWarmupScheduler

class BaseSSL(LightningModule):
    def __init__(self, backbone, config=None):
        super().__init__()

        self.backbone = backbone
        self.projection_head = None
        self.criterion = None

        if config != None:
            self.learning_rate = config.train_learning_rate
            self.weight_decay = config.weight_decay
            self.momentum = config.momentum
            self.batch_size = (config.train_batch_size*config.train_accumulate_grad_batches)/config.num_devices
            self.trust_coefficient = config.trust_coefficient
            self.clip_lr = config.clip_lr
            self.name = config.model_name

    def forward(self, x):
        features = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(features)

        output = {"features": features, "projection": z}
        return output

    def get_params(self):
        params, params_no_weight_decay = get_weight_decay_parameters(
            [self.backbone, self.projection_head]
        )
        return params, params_no_weight_decay
    
    def configure_optimizers(self):
        params, params_no_weight_decay = self.get_params()
        
        # Don't use weight decay for batch norm, bias parameters, and classification
        # head to improve performance.
        optimizer = LARS([
                            {"name": f"{self.name}", "params": params},
                            {
                                "name": f"{self.name}_no_weight_decay",
                                "params": params_no_weight_decay,
                                "weight_decay": 0.0,
                            },
                        ], 
                        lr=self.learning_rate * math.sqrt(self.batch_size * self.trainer.world_size),
                        momentum=self.momentum, 
                        weight_decay=self.weight_decay,
                        trust_coefficient=self.trust_coefficient,
                        clip_lr=self.clip_lr)
        
        scheduler = {
            "scheduler": CosineWarmupScheduler(
                optimizer=optimizer,
                warmup_epochs=int(self.trainer.estimated_stepping_batches / self.trainer.max_epochs * 10),
                max_epochs=self.trainer.estimated_stepping_batches,
            ),
            "interval": "step",
        }
        return [optimizer], [scheduler]
