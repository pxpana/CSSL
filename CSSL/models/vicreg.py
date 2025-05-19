import torch
from lightly import loss
from lightly.models.modules.heads import VICRegProjectionHead
from lightly.utils.debug import std_of_l2_normalized

from models.base_ssl import BaseSSL

class VICReg(BaseSSL):
    def __init__(self, backbone, config=None):
        super().__init__(backbone, config)

        self.projection_head = VICRegProjectionHead(
            input_dim=config.feature_dim, 
            hidden_dim=config.projection_hidden_dim, 
            output_dim=config.projection_output_dim,
            num_layers=config.projection_layers
        )
        self.criterion = loss.VICRegLoss(
            lambda_param=config.sim_loss_weight,
            mu_param=config.var_loss_weight,
            nu_param=config.cov_loss_weight,
            gather_distributed=config.gather_distributed,
            eps=config.epsilon
        )
    
    def training_step(self, batch, batch_index):
        images, _, _ = batch
        view0, view1 = images[:, 0], images[:, 1]

        outputs = self.forward(view0)
        feats0, z0 = outputs["features"], outputs["projection"]
        z1 = self.forward(view1)["projection"]

        loss = self.criterion(z0, z1)
        representation_std = std_of_l2_normalized(feats0)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("representation_std", representation_std, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
