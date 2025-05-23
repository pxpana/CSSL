import torch
from lightly import loss
from lightly.models.modules import SimCLRProjectionHead
from lightly.utils.debug import std_of_l2_normalized

from models.base_ssl import BaseSSL

class SimCLR(BaseSSL):
    def __init__(self, backbone, config=None):
        super().__init__(backbone, config)

        self.projection_head = SimCLRProjectionHead(
            input_dim=512, hidden_dim=config.hidden_dim, output_dim=config.output_dim
        )
        self.criterion = loss.NTXentLoss(
            temperature=config.temperature,
            gather_distributed=True,
        )
    
    def training_step(self, batch, batch_index):
        view0, view1 = batch

        outputs = self.forward(view0)
        feats0, z0 = outputs["features"], outputs["projection"]
        z1 = self.forward(view1)["projection"]
        
        loss = self.criterion(z0, z1)
        representation_std = std_of_l2_normalized(feats0)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("representation_std", representation_std, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
