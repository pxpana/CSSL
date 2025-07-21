import math
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from lightly.models.modules import BarlowTwinsProjectionHead
from lightly.models.utils import (
    batch_shuffle,
    batch_unshuffle,
    update_momentum,
)
from lightly.models.utils import get_weight_decay_parameters, deactivate_requires_grad
from lightly.utils.debug import std_of_l2_normalized
from lightly.utils.scheduler import CosineWarmupScheduler, cosine_schedule

from cssl.loss import BarlowTwinsLoss
from cssl.models.base_ssl import BaseSSL

class BarlowTwins(BaseSSL):
    def __init__(self, backbone, config=None, *args, **kwargs):
        super().__init__(backbone, config, *args, **kwargs)

        self.projection_head = BarlowTwinsProjectionHead(
            input_dim=config.feature_dim, 
            hidden_dim=config.projection_hidden_dim, 
            output_dim=config.projection_output_dim
        )

        self.criterion = BarlowTwinsLoss(
            lambda_param=config.loss["lambda_param"], 
            scale_loss=config.loss["scale_loss"]
        )

    def training_step(self, batch, batch_index):
        view0, view1 = batch

        z0 = self.forward(view0)["projection"]
        z1 = self.forward(view1)["projection"]
        loss = self.criterion(z0, z1)

        representation_std = std_of_l2_normalized((z0 + z1) / 2)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("representation_std", representation_std, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss