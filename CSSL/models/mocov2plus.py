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
from lightly.utils.scheduler import cosine_schedule
from models.base_ssl import BaseSSL

class MoCov2Plus(BaseSSL):
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
        self.criterion0 = loss.NTXentLoss(
            temperature=config.temperature,
            memory_bank_size=(config.queue_size, config.output_dim),
            gather_distributed=True,
        )
        self.criterion1 = loss.NTXentLoss(
            temperature=config.temperature,
            memory_bank_size=(config.queue_size, config.output_dim),
            gather_distributed=True,
        )

        self.start_value = config.momentum_encoder["base_tau"]
        self.end_value = config.momentum_encoder["final_tau"]

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
        view0, view1 = batch

        # Encode queries.
        outputs = self.forward(view0)
        query_features0, query_projections0 = outputs["features"], outputs["projection"]
        outputs = self.forward(view1)
        _, query_projections1 = outputs["features"], outputs["projection"]

        # Momentum update. This happens between query and key encoding, following the
        # original implementation from the authors:
        # https://github.com/facebookresearch/moco/blob/5a429c00bb6d4efdf511bf31b6f01e064bf929ab/moco/builder.py#L142
        momentum = cosine_schedule(
            step=self.trainer.global_step,
            max_steps=self.trainer.estimated_stepping_batches,
            start_value=self.start_value,
            end_value=self.end_value,
        )
        update_momentum(self.backbone, self.key_backbone, m=momentum)
        update_momentum(self.projection_head, self.key_projection_head, m=momentum)

        key_projections0 = self.forward_key_encoder(view0)
        key_projections1 = self.forward_key_encoder(view1)
        loss = (
            self.criterion0(query_projections0, key_projections1)
            + self.criterion1(query_projections1, key_projections0)
            ) / 2
        
        representation_std = std_of_l2_normalized(query_features0)
        
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("representation_std", representation_std, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    

