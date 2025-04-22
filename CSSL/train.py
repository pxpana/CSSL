import argparse
import yaml
import torch
from tqdm import tqdm
import pytorch_lightning as pl
from torchvision.models import resnet18
from utils import DataManager, get_model, get_data_loader, get_classifier
from continuum.metrics.logger import Logger as LOGGER

def main(args):

    data_manager = DataManager(
        dataset_name=args.dataset,
        shuffle=True,
        seed=1992,
        init_cls=10,
        increment=10
    )
    
    backbone = resnet18()
    backbone.fc = torch.nn.Identity()
    model, pretrain_transform = get_model(backbone, args)
    logger = LOGGER()

    for current_task in tqdm(range(1, data_manager.nb_tasks), desc="Training tasks", unit="task"):
        increment = data_manager.get_task_size(current_task)

        pretrain_train_dataloader, classifier_train_dataloader, test_dataloader = get_data_loader(
            data_manager=data_manager, 
            current_task=current_task,
            pretrain_transform=pretrain_transform,
            args=args
        )

        # Train the model
        trainer = pl.Trainer(
            max_epochs=args.train_epochs, 
            accelerator=args.device,
            devices=args.gpu_devices
        )
        #trainer.fit(model, pretrain_train_dataloader)

        classifier = get_classifier(
            model.backbone, 
            num_classes=increment*current_task,
            logger=logger,
            current_task=current_task-1,
            args=args
        )

        # Evaluate the model
        trainer = pl.Trainer(
            max_epochs=args.test_epochs, 
            accelerator=args.device,
            devices=args.gpu_devices
        )
        trainer.fit(classifier, classifier_train_dataloader, test_dataloader)
        logger.end_task()






if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CSSL models")

    parser.add_argument("--config", type=str, help="Name of config file")
    args, remaining_args = parser.parse_known_args()

    with open(f"config/{args.config}.yaml", 'r') as file:
        config = yaml.safe_load(file)
    parser.set_defaults(**config)            # Set argparse defaults from YAML
    args = parser.parse_args(remaining_args) # Re-parse args with YAML defaults

    main(args)