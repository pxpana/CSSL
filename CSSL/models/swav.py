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

        self.crop_counts = config.crop_counts

        # Use a queue for small batch sizes (<= 256).
        self.start_queue_at_epoch = config.start_queue_at_epoch
        self.queue_size = config.queue_size
        self.n_batches_in_queue = 15
        self.queues = torch.nn.ModuleList(
            [
                MemoryBankModule(
                    size=self.queue_size
                )
                for _ in range(self.crop_counts[0])
            ]
        )

    def training_step(self, batch, batch_index):
        # The dataloader returns a list of image crops where the
        # first few items are high resolution crops and the rest are low
        # resolution crops.
        multi_crops = batch

        # Normalize the prototypes so they are on the unit sphere.
        self.prototypes.normalize()

        # Forward pass through backbone and projection head.
        multi_crop_features, multi_crop_projections = [], []
        for crop in multi_crops:
            outputs = self.forward(crop)
            multi_crop_features.append(outputs["features"])
            outputs["projection"] = F.normalize(outputs["projection"])
            multi_crop_projections.append(outputs["projection"])

        # Get the queue projections and logits.
        queue_crop_logits = None
        with torch.no_grad():
            if self.current_epoch >= self.start_queue_at_epoch:
                # Start filling the queue.
                queue_crop_projections = self._update_queue(
                    projections=multi_crop_projections[: self.crop_counts[0]],
                    queues=self.queues,
                )
                if batch_index > self.n_batches_in_queue:
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
            high_resolution_outputs=multi_crop_logits[: self.crop_counts[0]],
            low_resolution_outputs=multi_crop_logits[self.crop_counts[0] :],
            queue_outputs=queue_crop_logits,
        )

        representation_std = std_of_l2_normalized((multi_crop_projections[0] + multi_crop_projections[1]) / 2)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("representation_std", representation_std, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    @torch.no_grad()
    def _update_queue(
        self,
        projections,
        queues,
    ):
        """Adds the high resolution projections to the queues and returns the queues."""

        if len(projections) != len(queues):
            raise ValueError(
                f"The number of queues ({len(queues)}) should be equal to the number of high "
                f"resolution inputs ({len(projections)})."
            )

        # Get the queue projections
        queue_projections = []
        for i in range(len(queues)):
            _, queue_proj = queues[i](projections[i], update=True)
            # Queue projections are in (num_ftrs X queue_length) shape, while the high res
            # projections are in (batch_size_per_device X num_ftrs). Swap the axes for interoperability.
            queue_proj = torch.permute(queue_proj, (1, 0))
            queue_projections.append(queue_proj)

        return queue_projections
    
    def get_params(self):
        params, params_no_weight_decay = get_weight_decay_parameters(
            [self.backbone, self.projection_head, self.prototypes]
        )
        return params, params_no_weight_decay
