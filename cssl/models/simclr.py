import torch
from lightly import loss
from lightly.models.modules import SimCLRProjectionHead
from lightly.utils.debug import std_of_l2_normalized

from cssl.models.base_ssl import BaseSSL

class SimCLR(BaseSSL):
    def __init__(self, backbone, config=None):
        super().__init__(backbone, config)

        self.projection_head = SimCLRProjectionHead(
            input_dim=config.feature_dim, 
            hidden_dim=config.hidden_dim, 
            output_dim=config.output_dim,
            num_layers=config.num_layers,
            batch_norm=config.projection_batchnorm,
        )
        self.criterion = loss.NTXentLoss(
            temperature=config.temperature,
            gather_distributed=True,
        )
    
    def training_step(self, batch, batch_index):
        view0, view1 = batch["train_loader"][0], batch["train_loader"][1]

        out0 = self.forward(view0)
        out1 = self.forward(view1)

        z0 = out0["projection"]
        z1 = out1["projection"]

        loss = self.criterion(z0, z1)
        representation_std = std_of_l2_normalized((z0 + z1) / 2)

        # Oline linear classifier training
        cls_loss = self.online_training_step(batch["online_loader"], batch_index)

        loss += cls_loss

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("representation_std", representation_std, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
