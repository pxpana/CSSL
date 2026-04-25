import yaml, argparse
from tqdm import tqdm

import pytorch_lightning as pl
from lightly.models.utils import deactivate_requires_grad, activate_requires_grad

from cssl.utils import DataManager, get_callbacks_logger, get_classifier
from cssl.utils.factory import get_backbone, get_model
from cssl.metrics.logger import get_loggers

class Trainer:
    def __init__(self, config_path):
        self.config = self.get_config(config_path)

        # Get dataset
        self.data_manager = DataManager(config=self.config)

        # Get Loggers
        self.loggers = get_loggers(self.config, self.data_manager)

        # Get model
        backbone = get_backbone(self.config.backbone, self.config.dataset)
        self.model = get_model(backbone, self.config, loggers=self.loggers)

        # Get plugins


    def fit(self):
         pass


    def pretrain(self):
        activate_requires_grad(self.model.backbone)

        for scenario_id in tqdm(self.config.seeds, desc="⏳ Runing Scenerio"):
            train_classifier_loader = self.data_manager.train_classifier_loader
            test_classifier_loader = self.data_manager.test_classifier_loader
            for task_id, pretrain_dataloader in tqdm(enumerate(self.data_manager.pretrain_dataloaders), desc=f"💡 Training tasks"):    

                    pretrain_callbacks, pretrain_wandb_logger = get_callbacks_logger(
                        self.config, 
                        training_type="pretrain", 
                        task_id=task_id, 
                        scenario_id=scenario_id,
                    )

                    trainer = pl.Trainer(
                        max_epochs=self.config.train_epochs, 
                        accelerator=self.config.accelerator,
                        devices=self.config.gpu_devices,
                        accumulate_grad_batches=self.config.train_accumulate_grad_batches,
                        callbacks=pretrain_callbacks,
                        logger=pretrain_wandb_logger,
                        strategy=self.config.strategy,
                        precision=self.config.precision,
                        sync_batchnorm=self.config.sync_batchnorm,
                        num_sanity_val_steps=0,
                    )

                    trainer.fit(
                        self.model, 
                        train_dataloaders=pretrain_dataloader, 
                        val_dataloaders=[train_classifier_loader, test_classifier_loader]
                    )

    def evaluate(self):
        deactivate_requires_grad(self.model.backbone)

        _, classifier_wandb_logger = get_callbacks_logger(
            self.config, 
            training_type="classifier", 
            task_id=task_id, 
            scenario_id=scenario_id, 
            project=self.config.wandb_project
        )
            
        linear_classifier = get_classifier(
            self.model.backbone, 
            num_classes=self.config.num_classes,
            logger=None,
            args=self.config
        )
            
        trainer = pl.Trainer(
            max_epochs=self.config.test_epochs, 
            accelerator=self.config.accelerator,
            devices=self.config.gpu_devices,
            enable_checkpointing=False,
            logger=classifier_wandb_logger,
            strategy=self.config.strategy,
            precision=self.config.precision,
            sync_batchnorm=self.config.sync_batchnorm,
        )
        trainer.fit(
             linear_classifier, 
             train_dataloaders=train_classifier_loader, val_dataloaders=test_classifier_loader)
        



    def setup(self): 
         pass
    
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
