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

class Logger(continuum_Logger):
    def __init__(
            self, 
            random_classifiers, 
            args, 
            test_classifier_loader,
            class_increment,
            list_keywords=["performance"], root_log=None,
        ):
        super().__init__(list_keywords=list_keywords, root_log=root_log)

        self.random_init_accuracies = get_random_init_accuracies(
            random_classifiers, 
            test_classifier_loader, 
            num_tasks=args.num_tasks, 
            device=args.accelerator,
            class_increment=class_increment,
            args=args,
        )

    @property
    @require_subset("test")
    def average_accuracy(self):
        all_preds, all_targets, task_ids = self._get_best_epochs(subset="test")
        return average_accuracy(all_preds, all_targets, task_ids)
    
    # @property
    # @require_subset("test")
    # def forgetting(self):
    #     all_preds, all_targets, task_ids = self._get_best_epochs(subset="test")
    #     return forgetting(all_preds, all_targets, task_ids, self.current_task)

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

def forgetting(all_preds, all_targets, all_tasks, current_task):
    """Measures the average forgetting.

    Reference:
    * Riemannian Walk for Incremental Learning: Understanding Forgetting and Intransigence
      Chaudhry et al. ECCV 2018

    See eq. 3.
    """
    T = current_task+1  # Number of seen tasks so far
    # TODO if we take in account zeroshot, we should take the max of all_tasks?
    if T <= 1:
        return 0.0

    f = 0.0
    for j in range(T-1):
        # Accuracy on task j after learning current task k
        a_kj = _get_R_ij(T-1, j, all_preds, all_targets, all_tasks)
        # Best previous accuracy on task j
        max_a_lj = max(
            _get_R_ij(l, j, all_preds, all_targets, all_tasks)
            for l in range(T)
            if l >= j
        )
        f += max_a_lj - a_kj  # We want this results to be as low as possible

    metric = f / (T-1)
    assert -1.0 <= metric <= 1.0, metric
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

def get_random_init_accuracies(random_classifiers, test_classifier_loader, num_tasks, device, class_increment, args):
    device = device.lower()
    device = "cuda" if device in ["gpu", "cuda"] else "cpu"

    with torch.no_grad():
        random_init_accuracies = [ [] for _ in range(num_tasks) ]

        for task_id in range(num_tasks):
            model = random_classifiers[task_id]
            model.to(device)
            model.eval()

            for batch in tqdm(test_classifier_loader, desc=f"Calculating random init accuracies for task {task_id+1}"):
                images, targets, task_ids = batch[0], batch[1], batch[2]
                predictions = model(images.to(device))
                _, predicted_labels = predictions.topk(1)

                predicted_labels = predicted_labels.flatten() + (class_increment*task_id)
                correct = predicted_labels == targets.to(device)

                for task_idx in range(num_tasks):
                    mask_task = task_idx == task_ids
                    mask_correct = correct[mask_task]
                    
                    random_init_accuracies[task_id].extend(mask_correct.cpu().tolist())
        random_init_accuracies = [np.mean(np.array(acc)) for acc in random_init_accuracies]

    print(f"Random init accuracies: {random_init_accuracies}")

    return np.array(random_init_accuracies)

def get_random_classifiers(num_tasks, class_increment, feature_dim, seed):

    random_classifiers = []
    for i in range(num_tasks):
        torch.manual_seed(seed*i)
        
        backbone = resnet18(pretrained=False)
        backbone.fc = torch.nn.Identity()
        backbone.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=2, bias=False)
        backbone.maxpool = torch.nn.Identity()

        model = nn.Sequential(
            backbone,
            nn.Linear(feature_dim, class_increment)
        )

        random_classifiers.append(model)

    return random_classifiers

    