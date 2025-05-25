import math
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from lightly import loss
from lightly.models.modules import SimSiamPredictionHead, SimSiamProjectionHead
from lightly.models.utils import (
    batch_shuffle,
    batch_unshuffle,
    update_momentum,
)
from lightly.models.utils import get_weight_decay_parameters, deactivate_requires_grad
from lightly.utils.debug import std_of_l2_normalized
from lightly.utils.scheduler import CosineWarmupScheduler, cosine_schedule
from models.base_ssl import BaseSSL

class SimSiam(BaseSSL):
    def __init__(self, backbone, config=None):
        super().__init__(backbone, config)

        self.projection_head = SimSiamProjectionHead(
            input_dim=config.feature_dim, 
            hidden_dim=config.projection_hidden_dim, 
            output_dim=config.output_dim
        )
        self.prediction_head = SimSiamPredictionHead(
            input_dim=config.output_dim, 
            hidden_dim=config.prediction_hidden_dim, 
            output_dim=config.output_dim
        )

        self.criterion = loss.NegativeCosineSimilarity()

    def forward(self, x):
        features = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(features)
        p = self.prediction_head(z)

        output = {"features": features, "projection": z, "prediction": p}
        return output
    
    def training_step(self, batch, batch_idx):
        x0, x1 = batch

        output = self.forward(x0)
        feats0, z0, p0 = output["features"], output["projection"], output["prediction"]
        output = self.forward(x1)
        _, z1, p1 = output["features"], output["projection"], output["prediction"]
        
        loss = (
            self.criterion(z0, p1) + self.criterion(z1, p0)
        ) / 2

        representation_std = std_of_l2_normalized(feats0)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("representation_std", representation_std, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def get_params(self):
        params, params_no_weight_decay = get_weight_decay_parameters(
            [self.backbone, self.projection_head, self.prediction_head]
        )
        return params, params_no_weight_decay
    
    

