import math
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule

from lightly import loss
from lightly.models.modules import SimCLRProjectionHead
from lightly.models.utils import get_weight_decay_parameters
from lightly.utils.lars import LARS
from lightly.utils.scheduler import CosineWarmupScheduler
from lightly.utils.debug import std_of_l2_normalized

class SimCLR(LightningModule):
    def __init__(self, backbone, config=None):
        super().__init__()

        self.backbone = backbone
        self.projection_head = SimCLRProjectionHead(
            input_dim=512, hidden_dim=2048, output_dim=128
        )
        
        self.criterion = loss.DCLWLoss()

        if config != None:
            self.learning_rate = config.train_learning_rate
            self.weight_decay = config.weight_decay
            self.momentum = config.momentum
            self.batch_size = (config.train_batch_size*config.train_accumulate_grad_batches)/config.num_devices

    def forward(self, x):
        features = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(features)

        output = {"feats": features, "z": z}
        return output
    
    def training_step(self, batch, batch_index):
        view0, view1 = batch[0], batch[1]

        z0 = self.forward(view0)["z"]
        z1 = self.forward(view1)["z"]
        loss = self.criterion(z0, z1)
        representation_std = (std_of_l2_normalized(z0) + std_of_l2_normalized(z1))/2

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("representation_std", representation_std, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    
    def configure_optimizers(self):
        params, params_no_weight_decay = get_weight_decay_parameters(
            [self.backbone, self.projection_head]
        )
        
        # Don't use weight decay for batch norm, bias parameters, and classification
        # head to improve performance.
        optimizer = LARS([
                            {"name": "simclr", "params": params},
                            {
                                "name": "simclr_no_weight_decay",
                                "params": params_no_weight_decay,
                                "weight_decay": 0.0,
                            },
                        ], 
                        lr=self.learning_rate * math.sqrt(self.batch_size * self.trainer.world_size),
                        #lr=self.learning_rate, 
                        momentum=self.momentum, 
                        weight_decay=self.weight_decay)
        
        scheduler = {
            "scheduler": CosineWarmupScheduler(
                optimizer=optimizer,
                warmup_epochs=int(self.trainer.estimated_stepping_batches / self.trainer.max_epochs * 10),
                max_epochs=self.trainer.estimated_stepping_batches,
            ),
            "interval": "step",
        }
        return [optimizer], [scheduler]
