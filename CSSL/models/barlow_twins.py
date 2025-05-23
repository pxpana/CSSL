import math
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from loss import BarlowTwinsLoss
from lightly.models.modules import BarlowTwinsProjectionHead
from lightly.models.utils import (
    batch_shuffle,
    batch_unshuffle,
    update_momentum,
)
from lightly.models.utils import get_weight_decay_parameters, deactivate_requires_grad
from lightly.utils.debug import std_of_l2_normalized
from lightly.utils.scheduler import CosineWarmupScheduler, cosine_schedule
from models.base_ssl import BaseSSL

class BarlowTwins(BaseSSL):
    def __init__(self, backbone, config=None):
        super().__init__(backbone, config)

        self.projection_head = BarlowTwinsProjectionHead(
            input_dim=config.feature_dim, 
            hidden_dim=config.projection_hidden_dim, 
            output_dim=config.projection_output_dim
        )

        self.criterion = BarlowTwinsLoss(
            lambda_param=config.lambda_param, 
            scale_loss=config.scale_loss
        )

    def training_step(self, batch, batch_index):
        view0, view1 = batch

        output0 = self.forward(view0)
        output1 = self.forward(view1)
        loss = self.criterion(output0["projection"], output1["projection"])
        representation_std = std_of_l2_normalized(output0["features"])

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("representation_std", representation_std, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss