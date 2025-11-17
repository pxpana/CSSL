import torch
from lightly.models.modules import SimCLRProjectionHead
from lightly.utils.debug import std_of_l2_normalized

from cssl.models.base_ssl import BaseSSL

class SimCLR(BaseSSL):
    def __init__(self, backbone, config=None, *args, **kwargs):
        super().__init__(backbone, config, *args, **kwargs)

        self.projection_head = SimCLRProjectionHead(
            input_dim=config.feature_dim, 
            hidden_dim=config.hidden_dim, 
            output_dim=config.output_dim,
            num_layers=config.num_layers,
            batch_norm=config.projection_batchnorm,
        )

        if config.loss["name"] == "dclw":
            from lightly.loss.dcl_loss import DCLWLoss
            self.criterion = DCLWLoss(
                temperature=config.loss["temperature"], 
                sigma=config.loss["sigma"]
            )
        elif config.loss["name"] == "ntxent":
            from lightly.loss import NTXentLoss
            self.criterion = NTXentLoss(
                temperature=config.loss["temperature"],
                gather_distributed=True,
            )

    def training_step(self, batch, batch_index):
        view0, view1 = batch[0], batch[1]
        batch_size = view0.shape[0]

        z0 = self.forward(view0)["projection"]
        z1 = self.forward(view1)["projection"]

        loss = self.criterion(z0, z1)
        representation_std = std_of_l2_normalized((z0 + z1) / 2)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size)
        self.log("representation_std", representation_std, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size)

        return loss


