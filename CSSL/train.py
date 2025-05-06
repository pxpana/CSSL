import argparse
import yaml
import torch
import wandb
from tqdm import tqdm
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from torchvision.models import resnet18
from utils import DataManager, get_model, get_classifier, get_callbacks_logger
from continuum.metrics.logger import Logger as LOGGER

def main(args):
    seeds = args.seeds
    for scenario_id in tqdm(seeds, desc="Scenerio"):

        seed_everything(scenario_id)

        backbone = resnet18()
        backbone.fc = torch.nn.Identity()
        model, pretrain_transform = get_model(backbone, args)
        logger = LOGGER()

        data_manager = DataManager(
            train_pretrain_transform=pretrain_transform,
            args=args,
        )

        train_pretrain_scenario, train_classifier_scenario, test_classifier_scenario = data_manager.get_scenerio(scenario_id)

        for task_id, train_pretrain_taskset in tqdm(enumerate(train_pretrain_scenario), desc="Training tasks"):
            train_classifier_taskset = train_classifier_scenario[:task_id+1]
            test_classifier_taskset = test_classifier_scenario[:task_id+1]
            train_pretrain_loader, train_classifier_loader, test_classifier_loader = data_manager.get_dataloader(train_pretrain_taskset, train_classifier_taskset, test_classifier_taskset, args)

            pretrain_callbacks, pretrain_wandb_logger = get_callbacks_logger(args, training_type="pretrain", task_id=task_id, scenario_id=scenario_id)
            _, classifier_wandb_logger = get_callbacks_logger(args, training_type="classifier", task_id=task_id, scenario_id=scenario_id)

            model, _ = get_model(model.backbone, args)

            '''
            Train the model
            '''
            trainer = pl.Trainer(
                max_epochs=args.train_epochs, 
                accelerator=args.device,
                devices=args.gpu_devices,
                accumulate_grad_batches=args.train_accumulate_grad_batches,
                callbacks=pretrain_callbacks,
                logger=pretrain_wandb_logger
            )
            trainer.fit(model, train_pretrain_loader)

            '''
            Evaluate the model
            '''
            classifier = get_classifier(
                model.backbone, 
                num_classes=10*(task_id+1),
                logger=logger,
                args=args
            )

            trainer = pl.Trainer(
                max_epochs=args.test_epochs, 
                accelerator=args.device,
                devices=args.gpu_devices,
                enable_checkpointing=False,
                logger=classifier_wandb_logger,
                strategy='ddp_find_unused_parameters_true'
            )
            trainer.fit(classifier, train_classifier_loader, test_classifier_loader)

            wandb.finish()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CSSL models")

    parser.add_argument("--config", type=str, help="Name of config file")
    args, remaining_args = parser.parse_known_args()

    with open(f"config/{args.config}.yaml", 'r') as file:
        config = yaml.safe_load(file)
    parser.set_defaults(**config)            # Set argparse defaults from YAML
    args = parser.parse_args(remaining_args) # Re-parse args with YAML defaults

    main(args)