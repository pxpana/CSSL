import os, yaml, argparse
from tqdm import tqdm
import wandb

import torch

import pytorch_lightning as pl

from lightly.models.utils import deactivate_requires_grad

from cssl.utils import DataManager, get_callbacks_logger
from cssl.utils.factory import get_backbone
from cssl.metrics.logger import get_loggers_classifier


class Evaluator:
    def __init__(self, config_path):
        self.config = self.get_config(config_path)
        
        # Get dataset
        self.data_manager = DataManager(config=self.config)

        # Get Loggers
        self.loggers = get_loggers_classifier(self.config, self.data_manager)
        
        backbone = get_backbone(self.config.backbone, self.config.dataset)
        deactivate_requires_grad(backbone)
        
        self.model = self.task(backbone, self.config.task)
        
        
    
    def evaluate(self):
        
        for scenario_id in tqdm(self.config.seeds, desc="⏳ Runing Scenerio"):
            train_classifier_loader = self.data_manager.train_classifier_loader
            test_classifier_loader = self.data_manager.test_classifier_loader
            
            for task_id, pretrain_dataloader in tqdm(enumerate(self.data_manager.pretrain_dataloaders), desc=f"💡 Training tasks"):
                
                self.model.backbone = self.load_checkpoint(self.model.backbone, scenario_id, task_id+1)
                
                pretrain_callbacks, pretrain_wandb_logger = get_callbacks_logger(
                    self.config, 
                    training_type="pretrain", 
                    task_id=task_id, 
                    scenario_id=scenario_id,
                    project="CSSL_Downstream"
                )

                trainer = pl.Trainer(
                    max_epochs=self.config.train_epochs, 
                    accelerator=self.config.accelerator,
                    devices=self.config.gpu_devices,
                    accumulate_grad_batches=self.config.train_accumulate_grad_batches,
                    callbacks=pretrain_callbacks,
                    #logger=pretrain_wandb_logger,
                    strategy=self.config.strategy,
                    precision=self.config.precision,
                    sync_batchnorm=self.config.sync_batchnorm,
                    num_sanity_val_steps=0,
                )

                trainer.fit(
                    self.model, 
                    train_dataloaders=train_classifier_loader, 
                    val_dataloaders=test_classifier_loader
                )
                
                # if self.config.wandb:
                #     wandb.finish()
         
        
        
    
    def load_checkpoint(self, backbone, scenario_id, task_id):
        plugin = "" if self.config.plugin=="" else f"_{self.config.plugin}"
        dirpath = f"checkpoints/{self.config.model.lower() }_{self.config.dataset}_{self.config.split_strategy}{plugin}"
        filename = f"scenario_{scenario_id}_task_{task_id}"
        checkpoint = torch.load(os.path.join(dirpath, f"{filename}.pth"), map_location="cpu")
        
        backbone.load_state_dict(checkpoint, strict=True)
        return backbone
        
    def task(self, backbone, task_type):
        if task_type == "classification":
            from cssl.tasks.classification import Classification
            model =  Classification(backbone=backbone, config=self.config, loggers=self.loggers)
            
        return model
    
    def get_config(self, path):
        with open(path, 'r') as file:
            config = yaml.safe_load(file)

        parser = argparse.ArgumentParser(description="Train CSSL models")
        for key, value in config.items():
            if isinstance(value, bool):
                parser.add_argument(f"--{key}", type=bool, default=value)
            else:
                parser.add_argument(f"--{key}", type=type(value), default=value)
        
        return parser.parse_args()