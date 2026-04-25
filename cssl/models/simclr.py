import math
import torch
from lightly.models.modules import SimCLRProjectionHead
from lightly.utils.debug import std_of_l2_normalized
from lightly.models.utils import get_weight_decay_parameters
from lightly.loss import NTXentLoss

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

        self.criterion = NTXentLoss(
            temperature=config.loss["temperature"],
            gather_distributed=True,
        )

    def forward(self, x):
        features = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(features)

        output = {"features": features, "projection": z}
        return output

    def training_step(self, batch, batch_index):
        view0, view1, targets = batch[0], batch[1], batch[2]
        batch_size = view0.shape[0]

        z0 = self.forward(view0)["projection"]
        z1 = self.forward(view1)["projection"]

        loss = self.criterion(z0, z1)
        z0 = torch.nn.functional.normalize(z0, dim=1)
        representation_std = std_of_l2_normalized(z0)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size)
        self.log("representation_std", representation_std, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size)

        if self.config.use_online_classifier:
            online_log = self.online_classifier.training_step(
                [torch.concat([z0, z1]).detach(), targets.repeat(2)], 
                batch_index
            )
            online_log['train_online_cls_loss'] = online_log.pop('loss')
            loss += online_log['train_online_cls_loss']
            self.log_dict(online_log, sync_dist=True, batch_size=len(targets))

        return loss
    
    def get_params(self):
        params, params_no_weight_decay = get_weight_decay_parameters(
            [self.backbone, self.projection_head]
        )
        return params, params_no_weight_decay
    
    def get_effective_lr(self) -> float:
        # Square root learning rate scaling improves performance for small
        # batch sizes (<=2048) and few training epochs (<=200). Alternatively,
        return self.config.optimizer["train_learning_rate"] * math.sqrt(self.config.train_batch_size * self.trainer.world_size)


