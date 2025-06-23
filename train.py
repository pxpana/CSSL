import os
import argparse
import yaml
from copy import deepcopy
import torch
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from torchvision.models import resnet18
from cssl.utils import DataManager, get_model, get_pretrain_transform, get_classifier, get_callbacks_logger, get_checkpoint
#from continuum.metrics.logger import Logger as LOGGER
from cssl.metrics.logger import Logger, get_random_classifiers
from cssl.dataset import ClassifierDataset, PretrainDataset
from lightly.models.utils import deactivate_requires_grad, activate_requires_grad

def main(args):
    seeds = args.seeds
    class_increment = int(args.num_classes / args.num_tasks)

    # initialize random classifiers equal to the number of tasks for FWT metric
    random_classifiers = get_random_classifiers(args.num_tasks, class_increment, args.feature_dim, seed=42)

    for scenario_id in tqdm(seeds, desc="Scenerio"):

        seed_everything(scenario_id)
        torch.set_float32_matmul_precision(args.set_float32_matmul_precision)

        backbone = resnet18(pretrained=False)
        backbone.fc = torch.nn.Identity()
        backbone.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=2, bias=False)
        backbone.maxpool = torch.nn.Identity()
        model = get_model(backbone, args)

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

        if not os.path.exists(f"logs/{args.model_name}_linear_{args.dataset}_{args.num_tasks}"):
            os.makedirs(f"logs/{args.model_name}_linear_{args.dataset}_{args.num_tasks}")
            os.makedirs(f"logs/{args.model_name}_knn_{args.dataset}_{args.num_tasks}")
            os.makedirs(f"logs/{args.model_name}_ncm_{args.dataset}_{args.num_tasks}")
        logger = Logger(
            random_classifiers=random_classifiers,
            args=args,
            test_classifier_loader=test_classifier_loader,
            class_increment=class_increment,
            list_keywords=["performance"],
            root_log=f"logs/{args.model_name}_linear_{args.dataset}_{args.num_tasks}"
        )

        knn_logger = deepcopy(logger)
        knn_logger.root_log = f"logs/{args.model_name}_knn_{args.dataset}_{args.num_tasks}"
        ncm_logger = deepcopy(logger)
        ncm_logger.root_log = f"logs/{args.model_name}_ncm_{args.dataset}_{args.num_tasks}"

        loggers = {
            "linear": logger,
            "knn": knn_logger,
            "ncm": ncm_logger
        }

        for task_id, train_dataset in tqdm(enumerate(data_manager.train_pretrain_scenario), desc="Training tasks"):
            print("TASK ID", task_id)

            pretrain_dataset = PretrainDataset(data=train_dataset, transform=data_manager.pretrain_transform)
            train_pretrain_loader = DataLoader(
                pretrain_dataset, 
                batch_size=args.train_batch_size, 
                num_workers=args.num_workers, 
                shuffle=True
            )
            online_classifier_loader = DataLoader(
                test_classifier_dataset, 
                batch_size=args.train_batch_size, 
                num_workers=args.num_workers, 
                shuffle=True
            )

            train_pretrain_loader = {"train_loader": train_pretrain_loader, "online_loader": online_classifier_loader}

            pretrain_callbacks, pretrain_wandb_logger = get_callbacks_logger(args, training_type="pretrain", task_id=task_id, scenario_id=scenario_id)
            _, classifier_wandb_logger = get_callbacks_logger(args, training_type="classifier", task_id=task_id, scenario_id=scenario_id)

            if task_id>0:
                model = get_model(model.backbone, args)

            activate_requires_grad(model.backbone)

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
            trainer.fit(model, train_pretrain_loader, test_classifier_loader)

            '''
            Evaluate the model 
            '''
            
            deactivate_requires_grad(model.backbone)

            classifiers = get_classifier(
                model.backbone, 
                num_classes=args.num_classes,
                loggers=loggers,
                classifier_type=["linear", "knn", "ncm"],
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
            trainer.fit(classifiers["linear"], train_classifier_loader, test_classifier_loader)

            # KNN CLASSIFIER

            trainer = pl.Trainer(
                max_epochs=1, 
                accelerator=args.accelerator,
                devices=args.gpu_devices,
                enable_checkpointing=False,
                strategy=args.strategy,
                precision=args.precision,
                sync_batchnorm=args.sync_batchnorm,
            )
            # trainer.validate(
            #     model=classifiers["knn"], 
            #     dataloaders=[train_classifier_loader, test_classifier_loader]
            # )

            # NCM CLASSIFIER

            trainer = pl.Trainer(
                max_epochs=1, 
                accelerator=args.accelerator,
                devices=args.gpu_devices,
                enable_checkpointing=False,
                strategy=args.strategy,
                precision=args.precision,
                sync_batchnorm=args.sync_batchnorm,
            )
            # trainer.validate(
            #     model=classifiers["ncm"], 
            #     dataloaders=[train_classifier_loader, test_classifier_loader]
            # )


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