import math
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from lightly.loss.swav_loss import SwaVLoss
from lightly.models.modules import SwaVProjectionHead, SwaVPrototypes
from lightly.models.modules.memory_bank import MemoryBankModule
from lightly.models.utils import get_weight_decay_parameters, deactivate_requires_grad
from lightly.utils.debug import std_of_l2_normalized
from lightly.utils.scheduler import CosineWarmupScheduler, cosine_schedule
from models.base_ssl import BaseSSL

CROP_COUNTS = (2, 6)

class SwAV(BaseSSL):
    def __init__(self, backbone, config=None):
        super().__init__(backbone, config)

        self.projection_head = SwaVProjectionHead(
            input_dim=config.feature_dim, 
            hidden_dim=config.projection_hidden_dim, 
            output_dim=config.projection_output_dim
        )
        self.prototypes = SwaVPrototypes(
            n_prototypes=config.num_prototypes, 
            n_steps_frozen_prototypes=config.freeze_prototypes_epochs
        )
        self.criterion = SwaVLoss(temperature=config.temperature)

        # Use a queue for small batch sizes (<= 256).
        self.start_queue_at_epoch = config.start_queue_at_epoch
        self.queue_size = config.queue_size
        self.queues = torch.nn.ModuleList(
            [
                MemoryBankModule(
                    size=self.queue_size
                )
                for _ in range(CROP_COUNTS[0])
            ]
        )

    def training_step(self, batch, batch_index):
        # The dataloader returns a list of image crops where the
        # first few items are high resolution crops and the rest are low
        # resolution crops.
        multi_crops, _, _ = batch
        print(multi_crops.shape)
        sdssd

        # Normalize the prototypes so they are on the unit sphere.
        self.prototypes.normalize()

        # Forward pass through backbone and projection head.
        multi_crop_features, multi_crop_projections = [], []
        for crops in multi_crops:
            ouput = self.forward(crops).flatten(start_dim=1)
            multi_crop_features.append(output["features"])
            multi_crop_projections.append(output["projection"])

        # Get the queue projections and logits.
        queue_crop_logits = None
        with torch.no_grad():
            if self.current_epoch >= self.start_queue_at_epoch:
                # Start filling the queue.
                queue_crop_projections = _update_queue(
                    projections=multi_crop_projections[: CROP_COUNTS[0]],
                    queues=self.queues,
                )
                if batch_idx > self.n_batches_in_queue:
                    # The queue is filled, so we can start using it.
                    queue_crop_logits = [
                        self.prototypes(projections, step=self.current_epoch)
                        for projections in queue_crop_projections
                    ]

        # Get the rest of the multi-crop logits.
        multi_crop_logits = [
            self.prototypes(projections, step=self.current_epoch)
            for projections in multi_crop_projections
        ]

        # Calculate the SwAV loss.
        loss = self.criterion(
            high_resolution_outputs=multi_crop_logits[: CROP_COUNTS[0]],
            low_resolution_outputs=multi_crop_logits[CROP_COUNTS[0] :],
            queue_outputs=queue_crop_logits,
        )

        representation_std = std_of_l2_normalized(output0["features"])

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("representation_std", representation_std, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
