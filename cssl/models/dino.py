import math
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from lightly import loss
from lightly.models.modules import DINOProjectionHead
from lightly.models.modules.memory_bank import MemoryBankModule
from lightly.models.utils import update_momentum
from lightly.utils.debug import std_of_l2_normalized
from lightly.utils.scheduler import CosineWarmupScheduler, cosine_schedule
from cssl.models.base_ssl import BaseSSL

class DINO(BaseSSL):
    def __init__(self, backbone, config=None, *args, **kwargs):
        super().__init__(backbone, config, *args, **kwargs)

        self.projection_head = DINOProjectionHead(
            input_dim=config.feature_dim, 
            hidden_dim=config.projection_hidden_dim,
            bottleneck_dim=64, 
            output_dim=config.projection_output_dim,
            freeze_last_layer=1
        )
        self.teacher_backbone = copy.deepcopy(backbone)
        self.teacher_head = DINOProjectionHead(
            input_dim=config.feature_dim, 
            hidden_dim=config.projection_hidden_dim,
            bottleneck_dim=64, 
            output_dim=config.projection_output_dim,
            freeze_last_layer=1
        )
        self.criterion = loss.DINOLoss(
            output_dim=config.projection_output_dim, 
            warmup_teacher_temp_epochs=5
        )

        self.start_value = config.momentum_encoder["base_tau"]
        self.end_value = config.momentum_encoder["final_tau"]

    def forward_teacher(self, x):
        y = self.teacher_backbone(x).flatten(start_dim=1)
        z = self.teacher_head(y)
        z = z.detach()
        return z

    def training_step(self, batch, batch_idx):
        global_views = batch[:2]

        # Momentum update teacher.
        momentum = cosine_schedule(
            step=self.trainer.global_step,
            max_steps=self.trainer.estimated_stepping_batches,
            start_value=self.start_value,
            end_value=self.end_value,
        )

        update_momentum(self.backbone, self.teacher_backbone, m=momentum)
        update_momentum(self.projection_head, self.teacher_head, m=momentum)

        student_outputs = [self.forward(view) for view in batch]
        student_projections = [out["projection"] for out in student_outputs]
        teacher_projections = [self.forward_teacher(view) for view in global_views]
        
        loss = self.criterion(teacher_projections, student_projections, epoch=self.current_epoch)
        representation_std = std_of_l2_normalized(student_outputs[0]["features"])

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("representation_std", representation_std, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def on_after_backward(self):
        self.projection_head.cancel_last_layer_gradients(current_epoch=self.current_epoch)

