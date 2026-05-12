import torch
import os
from pytorch_lightning.callbacks import Callback

class BackboneCheckpoint(Callback):
    def __init__(self, dirpath, filename):
        super().__init__()
        self.dirpath = dirpath
        self.filename = filename

    def on_train_epoch_end(self, trainer, pl_module):
        # Define the path
        epoch = trainer.current_epoch
        file_path = os.path.join(self.dirpath, f"{self.filename}.pth")
        
        # Extract and save just the backbone's state dict
        torch.save(pl_module.backbone.state_dict(), file_path)
