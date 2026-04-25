import torch.nn as nn

from lightly.models.utils import get_weight_decay_parameters, deactivate_requires_grad

class CaSSLe:
    def __init__(self, config, model, prev_backbone=None):
        self.config = config
        self.model = model
        self.prev_backbone = prev_backbone
        
        deactivate_requires_grad(prev_backbone)
        
        self.prediction_head = nn.Sequential(
            nn.Linear(config.feature_dim, config.prediction_hidden_dim),
            nn.BatchNorm1d(config.prediction_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.prediction_hidden_dim, config.feature_dim)
        )
        
    
    def forward(self, features):
        pred_features = self.prediction_head(features)
        prev_features = self.prev_backbone(features)
        
        output = {"pred_features": pred_features, "prev_features": prev_features}
        return output
    
    def training_step(self, batch, batch_index):
        ssl_loss = self.model.training_step(batch, batch_index)
        
        view0 = batch[0]
        features = self.model.backbone(view0).flatten(start_dim=1)
        
        pred_features = self.forward(features)["pred_features"]
        prev_features = self.forward(features)["prev_features"]
        
        cassle_loss = self.model.criterion(pred_features, prev_features)
        
        loss = ssl_loss + cassle_loss
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss