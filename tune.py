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
#from continuum.metrics.logger import Logger as LOGGER
from cssl.metrics.logger import Logger, get_random_classifiers, get_random_init_accuracies
from cssl.dataset import ClassifierDataset, PretrainDataset
from lightly.models.utils import deactivate_requires_grad, activate_requires_grad

import optuna

import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import SubsetRandomSampler, DataLoader

def tune(config):
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    study = optuna.create_study(
        study_name='my_optimization_study',
        storage=f'sqlite:///logs/{config.model_name}_{config.dataset}_{config.split_strategy}_tune.db',
        load_if_exists=True,
        direction='maximize', 
        pruner=pruner,
    )

    df = study.trials_dataframe()
    if len(df) != 0:
        num_complete = len(df[df["state"]=="COMPLETE"])
        print(f"Number of completed trials: {num_complete}")
    else:
        num_complete = 0
        print("No completed trials found.")

    if num_complete < config.num_trials:
        study.optimize(
            objective, 
            n_trials=int(config.num_trials),
        )

    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
        
    return study

def objective(trial):
    """Objective function for hyperparameter tuning."""
    # Set the hyperparameters
    config.optimizer["train_learning_rate"] = trial.suggest_float("learning_rate", 1e-5, 5e-1, log=True)
    config.optimizer["train_weight_decay"] = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)

    if "loss" in config:
        if "temperature" in config.loss:
            config.loss["temperature"] = trial.suggest_float("temperature", 0.01, 1.0, log=True)
        if "sigma" in config.loss:
            config.loss["sigma"] = trial.suggest_float("sigma", 0.01, 1.0, log=True)
        # if "lambda_param" in config.loss:
        #     config.loss["lambda_param"] = trial.suggest_float("lambda_param", 0.01, 1.0, log=True)
        # if "scale_loss" in config.loss:
        #     config.loss["scale_loss"] = trial.suggest_float("scale_loss", 0.01, 1.0, log=True)
        if "sim_loss_weight" in config.loss:
            config.loss["sim_loss_weight"] = trial.suggest_float("sim_loss_weight", 1.0, 100.0, log=True)
        if "var_loss_weight" in config.loss:
            config.loss["var_loss_weight"] = trial.suggest_float("var_loss_weight", 1.0, 100.0, log=True)
        if "cov_loss_weight" in config.loss:
            config.loss["cov_loss_weight"] = trial.suggest_float("cov_loss_weight", 0.01, 100.0, log=True)
        if "epsilon" in config.loss:
            config.loss["epsilon"] = trial.suggest_float("epsilon", 0.01, 1.0, log=True)

    # Train the model
    results = train(config)

    return results

def train(config):
    scenario_id = config.seeds[0]
    torch.set_float32_matmul_precision(config.set_float32_matmul_precision)

    data_manager = DataManager(
        args=config,
    )

    classifier_dataset = ClassifierDataset(data=data_manager.train_dataset, transform=data_manager.train_classifier_transform, tasks=data_manager.tasks)

    targets = classifier_dataset.targets

    indices = np.arange(len(classifier_dataset))
    train_idx, valid_idx = train_test_split(indices, test_size=0.2, shuffle=True, stratify=targets)

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_classifier_loader = DataLoader(
        classifier_dataset, 
        batch_size=config.test_batch_size, 
        num_workers=config.num_workers, 
        shuffle=False,
        sampler=train_sampler,
    )
    val_classifier_loader = DataLoader(
        classifier_dataset, 
        batch_size=config.test_batch_size, 
        num_workers=config.num_workers, 
        shuffle=False,
        sampler=valid_sampler,
    )


    random_classifiers = get_random_classifiers(config)
    random_init_accuracies = get_random_init_accuracies(
        random_classifiers,
        train_classifier_loader, 
        val_classifier_loader, 
        num_tasks=config.num_tasks, 
        device=config.accelerator,
        args=config,
    )

    logger = Logger(
        random_init_accuracies=random_init_accuracies["linear"],
        list_keywords=["performance"],
        root_log=None
    )

    knn_logger = deepcopy(logger)
    knn_logger.root_log = f"logs/{config.model_name}_knn_{config.dataset}_{config.num_tasks}"
    knn_logger.random_init_accuracies = random_init_accuracies["knn"]
    ncm_logger = deepcopy(logger)
    ncm_logger.root_log = f"logs/{config.model_name}_ncm_{config.dataset}_{config.num_tasks}"
    ncm_logger.random_init_accuracies = random_init_accuracies["ncm"]

    loggers = {
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
        config=config,  
        loggers=loggers
    )

    for task_id, train_dataset in tqdm(enumerate(data_manager.train_pretrain_scenario), desc="Training tasks"):
        print("TASK ID", task_id)

        pretrain_dataset = PretrainDataset(data=train_dataset, transform=data_manager.pretrain_transform)
        train_pretrain_loader = DataLoader(
            pretrain_dataset, 
            batch_size=config.train_batch_size, 
            num_workers=config.num_workers, 
            shuffle=True
        )

        '''
        Train the model
        '''
        trainer = pl.Trainer(
            max_epochs=config.train_epochs, 
            accelerator=config.accelerator,
            devices=config.gpu_devices,
            accumulate_grad_batches=config.train_accumulate_grad_batches,
            strategy=config.strategy,
            precision=config.precision,
            sync_batchnorm=config.sync_batchnorm,
            num_sanity_val_steps=0,
            check_val_every_n_epoch=config.check_val_every_n_epoch,
            enable_checkpointing=False,
            logger=False,
            callbacks=[TerminateOnNaN()],
        )
        trainer.fit(model, train_dataloaders=train_pretrain_loader, val_dataloaders=[train_classifier_loader, val_classifier_loader])

        if "KNN Ave. Accuracy" in trainer.callback_metrics:
            ave_accuracy = trainer.callback_metrics.get(f"KNN Ave. Accuracy").item()
        else:
            ave_accuracy = 0.0

    return ave_accuracy

from pytorch_lightning.callbacks import Callback

class TerminateOnNaN(Callback):
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # Check if loss is NaN (handle both dict and tensor outputs)
        loss = outputs.get('loss') if isinstance(outputs, dict) else outputs
        if torch.is_tensor(loss) and torch.isnan(loss).any():
            print("NaN detected in loss. Terminating training gracefully.")
            trainer.should_stop = True  # Stops training at the end of the current batch


if __name__ == "__main__":
    # First pass: just get the config file name
    initial_parser = argparse.ArgumentParser(description="Tune CSSL Models", add_help=False)
    initial_parser.add_argument("--config", type=str, help="Name of config file")
    initial_args, remaining_args = initial_parser.parse_known_args()
    
    # Load config file
    config_name = initial_args.config.lower()
    with open(f"config/model/{config_name}.yaml", 'r') as file:
        config = yaml.safe_load(file)
    
    # Main parser with all arguments
    parser = argparse.ArgumentParser(description="Tune CSSL Models")
    parser.add_argument("--num_trials", type=int, default=50, help="")
    parser.add_argument("--train_epochs", type=int, default=250, help="")
    parser.add_argument("--check_val_every_n_epoch", type=int, default=50, help="")
    parser.add_argument("--config", type=str, required=True, help="Name of config file")
    
    # Add all config parameters as optional arguments
    # Add config parameters, skipping duplicates
    fixed_args = {'num_trials', 'train_epochs', 'check_val_every_n_epoch'}
    for key, value in config.items():
        if key not in fixed_args:  # Skip already-added arguments
            if isinstance(value, bool):
                parser.add_argument(f"--{key}", action='store_true' if value else 'store_false')
            else:
                parser.add_argument(f"--{key}", type=type(value), default=value)
    
    # Parse all arguments
    config = parser.parse_args()

    console = Console()
    yaml_str = yaml.dump(config, sort_keys=False, default_flow_style=False, indent=2)
    console.print(Syntax(yaml_str, "yaml", theme="monokai", line_numbers=True))

    study = tune(config)
    optuna.visualization.plot_optimization_history(study)
    optuna.visualization.plot_param_importances(study)
    optuna.visualization.plot_parallel_coordinate(study)
    


