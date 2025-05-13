import math
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from lightly import loss
from lightly.models.modules import MoCoProjectionHead
from lightly.models.utils import (
    batch_shuffle,
    batch_unshuffle,
    update_momentum,
)

from lightly.utils.debug import std_of_l2_normalized
from models.base_ssl import BaseSSL

class MoCov2(BaseSSL):
    def __init__(self, backbone, config=None):
        super().__init__(backbone, config)

        self.projection_head = MoCoProjectionHead(
            input_dim=config.feature_dim, 
            hidden_dim=config.hidden_dim,
            output_dim=config.output_dim,
            num_layers=2
        )
        self.key_backbone = copy.deepcopy(self.backbone)
        self.key_projection_head = copy.deepcopy(self.projection_head)
        self.criterion = loss.NTXentLoss(
            temperature=config.temperature,
            memory_bank_size=(config.queue_size, config.output_dim),
            gather_distributed=True,
        )

    @torch.no_grad()
    def forward_key_encoder(self, x):
        x, shuffle = batch_shuffle(batch=x, distributed=self.trainer.num_devices > 1)
        features = self.key_backbone(x).flatten(start_dim=1)
        projections = self.key_projection_head(features)
        features = batch_unshuffle(
            batch=features,
            shuffle=shuffle,
            distributed=self.trainer.num_devices > 1,
        )
        projections = batch_unshuffle(
            batch=projections,
            shuffle=shuffle,
            distributed=self.trainer.num_devices > 1,
        )
        return projections

    def training_step(self, batch, batch_idx):
        images, _, _ = batch
        x_query, x_key = images[:, 0], images[:, 1]

        # Encode queries.
        outputs = self.forward(x_query)
        query_features, query_projections = outputs["features"], outputs["projection"]

        # Momentum update. This happens between query and key encoding, following the
        # original implementation from the authors:
        # https://github.com/facebookresearch/moco/blob/5a429c00bb6d4efdf511bf31b6f01e064bf929ab/moco/builder.py#L142
        update_momentum(self.backbone, self.key_backbone, m=0.999)
        update_momentum(self.projection_head, self.key_projection_head, m=0.999)

        # Encode keys.
        key_projections = self.forward_key_encoder(x_key)
        loss = self.criterion(query_projections, key_projections)
        
        representation_std = std_of_l2_normalized(query_features)
        
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("representation_std", representation_std, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    

