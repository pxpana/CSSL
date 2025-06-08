import os
import argparse
import yaml
import torch
import wandb
from tqdm import tqdm
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from torchvision.models import resnet18
from utils import DataManager, get_model, get_pretrain_transform, get_classifier, get_callbacks_logger, get_checkpoint
from continuum.metrics.logger import Logger as LOGGER
#from metrics.logger import Logger as LOGGER

def main(args):
    seeds = args.seeds
    class_increment = int(args.num_classes / args.num_tasks)

    for scenario_id in tqdm(seeds, desc="Scenerio"):

        seed_everything(scenario_id)
        torch.set_float32_matmul_precision(args.set_float32_matmul_precision)

        backbone = resnet18()
        backbone.fc = torch.nn.Identity()
        backbone.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=2, bias=False)
        backbone.maxpool = torch.nn.Identity()
        model = get_model(backbone, args)

        pretrain_transform = get_pretrain_transform(args)

        logger = LOGGER(
            list_keywords=["performance"]
        )

        data_manager = DataManager(
            args=args,
        )

        train_pretrain_scenario, train_classifier_scenario, test_classifier_scenario = data_manager.get_scenerio(scenario_id)

        for task_id, train_pretrain_taskset in tqdm(enumerate(train_pretrain_scenario), desc="Training tasks"):
            print("TASK ID", task_id)
            train_classifier_taskset = train_classifier_scenario[:task_id+1]
            test_classifier_taskset = test_classifier_scenario[:task_id+1]
            train_pretrain_loader, train_classifier_loader, test_classifier_loader = data_manager.get_dataloader(
                train_pretrain_taskset, 
                train_classifier_taskset, 
                test_classifier_taskset, 
                pretrain_transform,
                args
            )

            pretrain_callbacks, pretrain_wandb_logger = get_callbacks_logger(args, training_type="pretrain", task_id=task_id, scenario_id=scenario_id)
            _, classifier_wandb_logger = get_callbacks_logger(args, training_type="classifier", task_id=task_id, scenario_id=scenario_id)

            if task_id>0:
                model = get_model(model.backbone, args)

            '''
            Train the model
            '''
            trainer = pl.Trainer(
                max_epochs=args.train_epochs, 
                accelerator=args.accelerator,
                devices=args.gpu_devices,
                accumulate_grad_batches=args.train_accumulate_grad_batches,
                callbacks=pretrain_callbacks,
                logger=pretrain_wandb_logger,
                strategy=args.strategy,
                precision=args.precision,
                sync_batchnorm=args.sync_batchnorm,
            )
            trainer.fit(model, train_pretrain_loader)

            '''
            Evaluate the model
            '''
            classifier = get_classifier(
                model.backbone, 
                num_classes=class_increment*(task_id+1),
                logger=logger,
                args=args
            )

            trainer = pl.Trainer(
                max_epochs=args.test_epochs, 
                accelerator=args.accelerator,
                devices=args.gpu_devices,
                enable_checkpointing=False,
                logger=classifier_wandb_logger,
                strategy=args.strategy,
                precision=args.precision,
                sync_batchnorm=args.sync_batchnorm,
            )
            trainer.fit(classifier, train_classifier_loader, test_classifier_loader)

            wandb.finish()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CSSL models")

    parser.add_argument("--config", type=str, help="Name of config file")
    args, remaining_args = parser.parse_known_args()

    config_name = args.config.lower()
    with open(f"config/{config_name}.yaml", 'r') as file:
        config = yaml.safe_load(file)
    
    # Add all config parameters as optional arguments
    for key, value in config.items():
        if isinstance(value, bool):
            parser.add_argument(f"--{key}", type=bool, default=value)
        else:
            parser.add_argument(f"--{key}", type=type(value), default=value)
    
    # Parse again with full argument list
    args = parser.parse_args(remaining_args)
    
    main(args)