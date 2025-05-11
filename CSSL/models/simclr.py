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
        self.criterion = loss.DCLWLoss()
    
    def training_step(self, batch, batch_index):
        images, _, _ = batch
        view0, view1 = images[:, 0], images[:, 1]

        z0 = self.forward(view0)["z"]
        z1 = self.forward(view1)["z"]
        loss = self.criterion(z0, z1)
        representation_std = (std_of_l2_normalized(z0) + std_of_l2_normalized(z1))/2

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("representation_std", representation_std, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
