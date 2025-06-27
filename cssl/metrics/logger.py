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

from cssl.models import LinearClassifier
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

def get_random_init_accuracies(random_classifiers, train_classifier_loader, test_classifier_loader, num_tasks, device, args):
    device = device.lower()
    device = "cuda" if device in ["gpu", "cuda"] else "cpu"

    with torch.no_grad():
        random_init_accuracies = {
            "linear": [],
            "knn": [],
            "ncm": []
        }

        linear_classifier = random_classifiers["linear"]
        knn_classifier = random_classifiers["knn"]
        ncm_classifier = random_classifiers["ncm"]

        print("[bold magenta]LINEAR: Evaluating random initialization accuracies...[/bold magenta]")
        trainer = pl.Trainer(
            max_epochs=1, 
            accelerator=args.accelerator,
            devices=args.gpu_devices,
            enable_checkpointing=False,
            strategy=args.strategy,
            precision=args.precision,
            sync_batchnorm=args.sync_batchnorm,
            logger=False
        )
        trainer.validate(linear_classifier, test_classifier_loader)
        results = trainer.callback_metrics
        random_init_accuracies["linear"] = [
            results[f"Linear val_acc1_task{task_id+1}"].item() for task_id in range(num_tasks)
        ]

        print("[bold magenta]KNN: Evaluating random initialization accuracies...[/bold magenta]")
        trainer = pl.Trainer(
            max_epochs=1, 
            accelerator=args.accelerator,
            devices=args.gpu_devices,
            enable_checkpointing=False,
            strategy=args.strategy,
            precision=args.precision,
            sync_batchnorm=args.sync_batchnorm,
            logger=False
        )
        trainer.validate(knn_classifier, [train_classifier_loader, test_classifier_loader])
        results = trainer.callback_metrics
        random_init_accuracies["knn"] = [
            results[f"KNN val_acc1_task{task_id+1}"].item() for task_id in range(num_tasks)
        ]

        print("[bold magenta]NCM: Evaluating random initialization accuracies...[/bold magenta]")
        trainer = pl.Trainer(
            max_epochs=1, 
            accelerator=args.accelerator,
            devices=args.gpu_devices,
            enable_checkpointing=False,
            strategy=args.strategy,
            precision=args.precision,
            sync_batchnorm=args.sync_batchnorm,
            logger=False
        )
        trainer.validate(ncm_classifier, [train_classifier_loader, test_classifier_loader])
        results = trainer.callback_metrics
        random_init_accuracies["ncm"] = [
            results[f"NCM val_acc1_task{task_id+1}"].item() for task_id in range(num_tasks)
        ]

    return random_init_accuracies

def get_random_classifiers(num_tasks, num_classes, feature_dim):

    random_classifiers = {"linear": [], "knn": [], "ncm": []}
        
    backbone = resnet18(pretrained=False)
    backbone.fc = torch.nn.Identity()
    backbone.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=2, bias=False)
    backbone.maxpool = torch.nn.Identity()

    linear_classifier = LinearClassifier(
        model=backbone,
        batch_size_per_device=None,
        lr=None,
        feature_dim=feature_dim,
        num_classes=num_classes,
        num_tasks=num_tasks,
    )

    knn_classifier = KNNClassifier(
        model=backbone,
        num_classes=num_classes,
        knn_k=200,
        knn_t=0.1,
        num_tasks=num_tasks,
    )

    ncm_classifier = NCMClassifier(
        model=backbone,
        num_classes=num_classes,
        num_tasks=num_tasks,
    )

    random_classifiers["linear"] = linear_classifier
    random_classifiers["knn"] = knn_classifier
    random_classifiers["ncm"] = ncm_classifier

    return random_classifiers

    