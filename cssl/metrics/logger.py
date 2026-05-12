import os
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from continuum.metrics.metrics import _get_R_ij 
from continuum.metrics.logger import Logger as continuum_Logger
from continuum.metrics.utils import require_subset

from torchvision.models import resnet18
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from lightly.models.utils import deactivate_requires_grad
from lightly.utils.benchmarking import LinearClassifier

from cssl.models import OnlineLinearClassifier
from cssl.models import KNNClassifier
from cssl.models import NCMClassifier

from rich import print

class Logger(continuum_Logger):
    def __init__(
            self, 
            random_init_accuracies, 
            list_keywords=["performance"], root_log=None,
        ):
        super().__init__(list_keywords=list_keywords, root_log=root_log)

        self.random_init_accuracies = random_init_accuracies

    @property
    @require_subset("test")
    def average_accuracy(self):
        all_preds, all_targets, task_ids = self._get_best_epochs(subset="test")
        return average_accuracy(all_preds, all_targets, task_ids)

    @property
    @require_subset("test")
    def forward_transfer(self):
        all_preds, all_targets, task_ids = self._get_best_epochs(subset="test")
        return forward_transfer(all_preds, all_targets, task_ids, self.current_task, self.random_init_accuracies)
    
    def end_task(self):
        if self.root_log is not None:
            self._save()
        self.current_task += 1
        self.current_epoch = 0
        self._update_dict_architecture(update_task=True)
    
    def _save(self):
        import pickle as pkl

        filename = f"logger.pkl"
        filename = os.path.join(self.root_log, filename)
        with open(filename, "wb") as f:
            pkl.dump(self.logger_dict, f, pkl.HIGHEST_PROTOCOL)

def average_accuracy(all_preds, all_targets, all_tasks):
    T = len(all_preds)  # Number of seen tasks so far
    # TODO if we take in account zeroshot, we should take the max of all_tasks?
    A = 0.0

    for j in range(T):
        A += _get_R_ij(T-1, j, all_preds, all_targets, all_tasks)

    metric = A / T
    assert 0.0 <= metric <= 1.0, metric
    return metric

def forward_transfer(all_preds, all_targets, all_tasks, current_task, random_init_accuracies):
    """Measures the influence that learning a task has on the performance of future tasks.

    Reference:
    * Gradient Episodic Memory for Continual Learning
      Lopez-paz & ranzato, NeurIPS 2017

    :param all_preds: All predicted labels up to now.
    :param all_targets: All targets up to now.
    :param all_tasks: All task ids up to now.
    :return: a float metric between 0 and 1.
    """
    T = current_task+1  # Number of seen tasks so far
    # TODO if we take in account zeroshot, we should take the max of all_tasks?
    if T <= 1:
        return 0.0

    fwt = 0.0
    for i in range(1, T):
        # in GEM, they sum over R_{i-1,i} - b_i, where b_i is accuracy at initialization, we ignore b_i here.
        # NB: to get the same forward transfer as GEM the result should reduced by  "1/(T-1) sum_i b_i"
        fwt += _get_R_ij(i - 1, i, all_preds, all_targets, all_tasks) - random_init_accuracies[i]

    metric = fwt / (T-1)
    assert -1.0 <= metric <= 1.0, metric
    return metric 

def get_random_init_accuracies(config, data_manager, random_classifiers):
    with torch.no_grad():
        random_init_accuracies = {}

        if config.use_online_classifier:
            print("[bold magenta]LINEAR: Evaluating random initialization accuracies...[/bold magenta]")
            linear_classifier = random_classifiers["online_linear"]
        
            trainer = pl.Trainer(
                max_epochs=config.test_epochs, 
                accelerator=config.accelerator,
                devices=config.gpu_devices,
                enable_checkpointing=False,
                strategy=config.strategy,
                precision=config.precision,
                sync_batchnorm=config.sync_batchnorm,
                logger=False
            )
            trainer.fit(linear_classifier, data_manager.train_classifier_loader, data_manager.test_classifier_loader)
            results = trainer.callback_metrics
            random_init_accuracies["online_linear"] = [
                results[f"OnlineLinearClassifier Task {task_id+1} Data"].item() for task_id in range(config.num_tasks)
            ]

        if config.use_knn_classifier:
            print("[bold magenta]KNN: Evaluating random initialization accuracies...[/bold magenta]")
            knn_classifier = random_classifiers["knn"]
            
            trainer = pl.Trainer(
                max_epochs=1, 
                accelerator=config.accelerator,
                devices=config.gpu_devices,
                enable_checkpointing=False,
                strategy=config.strategy,
                precision=config.precision,
                sync_batchnorm=config.sync_batchnorm,
                logger=False
            )
            trainer.validate(knn_classifier, [data_manager.train_classifier_loader, data_manager.test_classifier_loader])
            results = trainer.callback_metrics
            random_init_accuracies["knn"] = [
                results[f"KNN Task {task_id+1} Data"].item() for task_id in range(config.num_tasks)
            ]

        if config.use_ncm_classifier:
            print("[bold magenta]NCM: Evaluating random initialization accuracies...[/bold magenta]")
            ncm_classifier = random_classifiers["ncm"]
            
            trainer = pl.Trainer(
                max_epochs=1, 
                accelerator=config.accelerator,
                devices=config.gpu_devices,
                enable_checkpointing=False,
                strategy=config.strategy,
                precision=config.precision,
                sync_batchnorm=config.sync_batchnorm,
                logger=False
            )
            trainer.validate(ncm_classifier, [data_manager.train_classifier_loader, data_manager.test_classifier_loader])
            results = trainer.callback_metrics
            random_init_accuracies["ncm"] = [
                results[f"NCM Task {task_id+1} Data"].item() for task_id in range(config.num_tasks)
            ]

    return random_init_accuracies

def get_random_classifiers(config):
    backbone = resnet18(pretrained=False)
    backbone.fc = torch.nn.Identity()
    backbone.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=2, bias=False)
    backbone.maxpool = torch.nn.Identity()
    deactivate_requires_grad(backbone)
    
    random_classifiers = {}
    if config.use_online_classifier:
        linear_classifier = OnlineLinearClassifier(
            backbone=backbone,
            batch_size_per_device=config.test_batch_size,
            lr=config.optimizer["classifier_learning_rate"],
            feature_dim=config.feature_dim,
            num_classes=config.num_classes,
            num_tasks=config.num_tasks,
            config=config
        )
        random_classifiers["online_linear"] = linear_classifier
    if config.use_knn_classifier:
        knn_classifier = KNNClassifier(
            model=backbone,
            num_classes=config.num_classes,
            knn_k=config.knn_neighbours,
            knn_t=config.knn_temperature,
            num_tasks=config.num_tasks,
            config=config
        )
        random_classifiers["knn"] = knn_classifier
    if config.use_ncm_classifier:
        ncm_classifier = NCMClassifier(
            model=backbone,
            num_classes=config.num_classes,
            num_tasks=config.num_tasks,
            config=config
        )
        random_classifiers["ncm"] = ncm_classifier

    return random_classifiers

def get_loggers(config, data_manager):
    random_classifiers = get_random_classifiers(config)
    random_init_accuracies = get_random_init_accuracies(
        config,
        data_manager,
        random_classifiers,
    )

    plugin = "" if config.plugin=="" else f"_{config.plugin}"
    if not os.path.exists(f"logs/{config.model}_online_linear_{config.dataset}{plugin}_{config.num_tasks}"):
        os.makedirs(f"logs/{config.model}_online_linear_{config.dataset}{plugin}_{config.num_tasks}")
        os.makedirs(f"logs/{config.model}_knn_{config.dataset}{plugin}_{config.num_tasks}")
        os.makedirs(f"logs/{config.model}_ncm_{config.dataset}{plugin}_{config.num_tasks}")

    loggers = {}

    if config.use_online_classifier:
        classifier_logger = Logger(
            random_init_accuracies=random_init_accuracies["online_linear"],
            list_keywords=["performance"],
            root_log=f"logs/{config.model}_online_linear_{config.dataset}{plugin}_{config.num_tasks}"
        )
        loggers["online_linear"] = classifier_logger
    if config.use_knn_classifier:
        classifier_logger = Logger(
            random_init_accuracies=random_init_accuracies["knn"],
            list_keywords=["performance"],
            root_log=f"logs/{config.model}_knn_{config.dataset}{plugin}_{config.num_tasks}"
        )
        loggers["knn"] = classifier_logger
    if config.use_ncm_classifier:
        classifier_logger = Logger(
            random_init_accuracies=random_init_accuracies["ncm"],
            list_keywords=["performance"],
            root_log=f"logs/{config.model}_ncm_{config.dataset}{plugin}_{config.num_tasks}"
        )
        loggers["ncm"] = classifier_logger

    return loggers   

def get_loggers_classifier(config, data_manager):
    backbone = resnet18(pretrained=False)
    backbone.fc = torch.nn.Identity()
    backbone.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=2, bias=False)
    backbone.maxpool = torch.nn.Identity()
    deactivate_requires_grad(backbone)
    
    plugin = "" if config.plugin=="" else f"_{config.plugin}"
    if not os.path.exists(f"logs/{config.model}_linear_{config.dataset}{plugin}_{config.num_tasks}"):
        os.makedirs(f"logs/{config.model}_linear_{config.dataset}{plugin}_{config.num_tasks}")
    
    linear_classifier = OnlineLinearClassifier(
        backbone=backbone,
        batch_size_per_device=config.test_batch_size,
        lr=config.optimizer["classifier_learning_rate"],
        feature_dim=config.feature_dim,
        num_classes=config.num_classes,
        num_tasks=config.num_tasks,
        config=config
    )
    
    print("[bold magenta]LINEAR: Evaluating random initialization accuracies...[/bold magenta]")
    trainer = pl.Trainer(
        max_epochs=config.test_epochs, 
        accelerator=config.accelerator,
        devices=config.gpu_devices,
        enable_checkpointing=False,
        strategy=config.strategy,
        precision=config.precision,
        sync_batchnorm=config.sync_batchnorm,
        logger=False
    )
    trainer.fit(linear_classifier, data_manager.train_classifier_loader, data_manager.test_classifier_loader)
    results = trainer.callback_metrics
    random_init_accuracies = [
        results[f"OnlineLinearClassifier Task {task_id+1} Data"].item() for task_id in range(config.num_tasks)
    ]
    classifier_logger = Logger(
        random_init_accuracies=random_init_accuracies,
        list_keywords=["performance"],
        root_log=f"logs/{config.model}_online_linear_{config.dataset}{plugin}_{config.num_tasks}"
    )
    
    return classifier_logger
                             


    