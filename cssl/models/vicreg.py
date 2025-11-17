import torch
from lightly import loss
from lightly.models.modules.heads import VICRegProjectionHead
from lightly.utils.debug import std_of_l2_normalized

from cssl.models.base_ssl import BaseSSL

class VICReg(BaseSSL):
    def __init__(self, backbone, config=None, *args, **kwargs):
        super().__init__(backbone, config, *args, **kwargs)

        self.projection_head = VICRegProjectionHead(
            input_dim=config.feature_dim, 
            hidden_dim=config.projection_hidden_dim, 
            output_dim=config.projection_output_dim,
            num_layers=config.projection_layers
        )
        self.criterion = loss.VICRegLoss(
            lambda_param=config.loss["sim_loss_weight"],
            mu_param=config.loss["var_loss_weight"],
            nu_param=config.loss["cov_loss_weight"],
            gather_distributed=config.gather_distributed,
            eps=config.loss["epsilon"]
        )
    
    def training_step(self, batch, batch_index):
        view0, view1 = batch
        batch_size = view0.shape[0]

        z0 = self.forward(view0)["projection"]
        z1 = self.forward(view1)["projection"]

        loss = self.criterion(z0, z1)
        representation_std = std_of_l2_normalized((z0 + z1) / 2)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size)
        self.log("representation_std", representation_std, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size)
        return loss
