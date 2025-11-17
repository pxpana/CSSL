import os
import argparse
import yaml
from rich.syntax import Syntax
from rich.console import Console

from copy import deepcopy
import torch
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from torchvision.models import resnet18
from cssl.utils import DataManager, get_model, get_classifier, get_callbacks_logger
from cssl.metrics.logger import Logger, get_random_classifiers, get_random_init_accuracies
from cssl.dataset import ClassifierDataset, PretrainDataset, BufferDataset
from lightly.models.utils import deactivate_requires_grad, activate_requires_grad

def main(args):
    seeds = args.seeds

    # initialize random classifiers equal to the number of tasks for FWT metric
    for scenario_id in tqdm(seeds, desc="Scenario"):

        seed_everything(scenario_id)
        torch.set_float32_matmul_precision(args.set_float32_matmul_precision)

        data_manager = DataManager(
            args=args,
        )

        train_classifier_dataset = ClassifierDataset(data=data_manager.train_dataset, transform=data_manager.train_classifier_transform, tasks=data_manager.tasks)
        train_classifier_loader = DataLoader(
            train_classifier_dataset, 
            batch_size=args.test_batch_size, 
            num_workers=args.num_workers, 
            shuffle=True
        )
        test_classifier_dataset = ClassifierDataset(data=data_manager.test_dataset, transform=data_manager.test_classifier_transform, tasks=data_manager.tasks)
        test_classifier_loader = DataLoader(
            test_classifier_dataset, 
            batch_size=args.test_batch_size, 
            num_workers=args.num_workers, 
            shuffle=False
        )

        random_classifiers = get_random_classifiers(args)
        random_init_accuracies = get_random_init_accuracies(
            random_classifiers,
            train_classifier_loader, 
            test_classifier_loader, 
            num_tasks=args.num_tasks, 
            device=args.accelerator,
            args=args,
        )

        plugin = "" if args.plugin=="" else f"_{args.plugin}"
        if not os.path.exists(f"logs/{args.model_name}_linear_{args.dataset}{plugin}_{args.num_tasks}"):
            os.makedirs(f"logs/{args.model_name}_linear_{args.dataset}{plugin}_{args.num_tasks}")
            os.makedirs(f"logs/{args.model_name}_knn_{args.dataset}{plugin}_{args.num_tasks}")
            os.makedirs(f"logs/{args.model_name}_ncm_{args.dataset}{plugin}_{args.num_tasks}")
        logger = Logger(
            random_init_accuracies=random_init_accuracies["linear"],
            list_keywords=["performance"],
            root_log=f"logs/{args.model_name}_linear_{args.dataset}_{args.num_tasks}"
        )

        knn_logger = deepcopy(logger)
        knn_logger.root_log = f"logs/{args.model_name}_knn_{args.dataset}{plugin}_{args.num_tasks}"
        knn_logger.random_init_accuracies = random_init_accuracies["knn"]
        ncm_logger = deepcopy(logger)
        ncm_logger.root_log = f"logs/{args.model_name}_ncm_{args.dataset}{plugin}_{args.num_tasks}"
        ncm_logger.random_init_accuracies = random_init_accuracies["ncm"]

        loggers = {
            "linear": logger,
            "knn": knn_logger,
            "ncm": ncm_logger
        }

        # Initialize the model
        backbone = resnet18(pretrained=False)
        backbone.fc = torch.nn.Identity()
        backbone.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=2, bias=False)
        backbone.maxpool = torch.nn.Identity()
        model = get_model(
            backbone=backbone, 
            config=args,  
            loggers=loggers
        )

        for task_id, train_dataset in tqdm(enumerate(data_manager.train_pretrain_scenario), desc="Training tasks"):
            print("TASK ID", task_id)

            if args.plugin in ["experience_replay", "dark_experience_replay"]:
                pretrain_dataset = BufferDataset(
                    data=train_dataset, 
                    transform=data_manager.pretrain_transform,
                    task_id=task_id
                )
            else:
                pretrain_dataset = PretrainDataset(
                    data=train_dataset, 
                    transform=data_manager.pretrain_transform
                )
            train_pretrain_loader = DataLoader(
                pretrain_dataset, 
                batch_size=args.train_batch_size, 
                num_workers=args.num_workers, 
                shuffle=True
            )

            pretrain_callbacks, pretrain_wandb_logger = get_callbacks_logger(
                args, 
                training_type="pretrain", 
                task_id=task_id, 
                scenario_id=scenario_id, 
                project=args.wandb_project
            )

            '''
            Train the model
            '''

            activate_requires_grad(model.backbone)
            
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
                num_sanity_val_steps=0,
            )
            trainer.fit(model, train_dataloaders=train_pretrain_loader, val_dataloaders=[train_classifier_loader, test_classifier_loader])

            '''
            Evaluate the model 
            '''
            
            deactivate_requires_grad(model.backbone)

            _, classifier_wandb_logger = get_callbacks_logger(args, training_type="classifier", task_id=task_id, scenario_id=scenario_id, project=args.wandb_project)

            linear_classifier = get_classifier(
                model.backbone, 
                num_classes=args.num_classes,
                logger=loggers["linear"],
                args=args
            )

            # LINEAR CLASSIFIER

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
            trainer.fit(linear_classifier, train_dataloaders = train_classifier_loader, val_dataloaders=test_classifier_loader)


            if args.wandb:
                wandb.finish()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CSSL models")

    parser.add_argument("--config", type=str, help="Name of config file")
    args, remaining_args = parser.parse_known_args()


    '''
        Load Model Config File
    '''
    config_name = args.config.lower()
    with open(f"config/model/{config_name}.yaml", 'r') as file:
        config = yaml.safe_load(file)
    
    # Add all config parameters as optional arguments
    for key, value in config.items():
        if isinstance(value, bool):
            parser.add_argument(f"--{key}", type=bool, default=value)
        else:
            parser.add_argument(f"--{key}", type=type(value), default=value)
    
    # Parse again with full argument list
    args = parser.parse_args(remaining_args)

    console = Console()
    yaml_str = yaml.dump(config, sort_keys=False, default_flow_style=False, indent=2)

    '''
        Load Plugin Config File
    '''
    if args.plugin != "":
        with open(f"config/plugin/{args.plugin.lower()}.yaml", 'r') as file:
            plugin_config = yaml.safe_load(file)
        for key, value in plugin_config.items():
            if isinstance(value, bool):
                parser.add_argument(f"--{key}", type=bool, default=value)
            else:
                parser.add_argument(f"--{key}", type=type(value), default=value)
        args = parser.parse_args(remaining_args)
        yaml_str_plugin = yaml.dump(plugin_config, sort_keys=False, default_flow_style=False, indent=2)
        yaml_str = yaml_str + "\n# Plugin parameters\n" + yaml_str_plugin

    console.print(Syntax(yaml_str, "yaml", theme="monokai", line_numbers=True))
    
    main(args)