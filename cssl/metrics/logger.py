import os
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from continuum.metrics.metrics import _get_R_ij 
from continuum.metrics.logger import Logger as continuum_Logger
from continuum.metrics.utils import require_subset

class Logger(continuum_Logger):
    def __init__(
            self, 
            backbone, 
            args, 
            test_dataloder,
            list_keywords=["performance"], root_log=None,
        ):
        super().__init__(list_keywords=list_keywords, root_log=root_log)

        model = nn.Sequential(
            backbone,
            nn.Linear(args.feature_dim, args.num_classes)
        )

        self.random_init_accuracies = get_random_init_accuracies(
            model, 
            test_dataloder, 
            num_tasks=args.num_tasks, 
            device=args.accelerator
        )

    @property
    @require_subset("test")
    def forward_transfer(self):
        all_preds, all_targets, task_ids = self._get_best_epochs(subset="test")
        return forward_transfer(all_preds, all_targets, task_ids, self.random_init_accuracies)
    
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

def forward_transfer(all_preds, all_targets, all_tasks, random_init_accuracies):
    """Measures the influence that learning a task has on the performance of future tasks.

    Reference:
    * Gradient Episodic Memory for Continual Learning
      Lopez-paz & ranzato, NeurIPS 2017

    :param all_preds: All predicted labels up to now.
    :param all_targets: All targets up to now.
    :param all_tasks: All task ids up to now.
    :return: a float metric between 0 and 1.
    """
    T = len(all_preds)  # Number of seen tasks so far
    # TODO if we take in account zeroshot, we should take the max of all_tasks?
    if T <= 1:
        return 0.0

    fwt = 0.0
    for i in range(2, T):
        # in GEM, they sum over R_{i-1,i} - b_i, where b_i is accuracy at initialization, we ignore b_i here.
        # NB: to get the same forward transfer as GEM the result should reduced by  "1/(T-1) sum_i b_i"
        fwt += _get_R_ij(i - 1, i, all_preds, all_targets, all_tasks) - random_init_accuracies[i]

    metric = fwt / (T - 1)
    assert -1.0 <= metric <= 1.0, metric
    return metric

def get_random_init_accuracies(model, test_dataloder, num_tasks, device):
    device = device.lower()
    device = "cuda" if device in ["gpu", "cuda"] else "cpu"
    model.to(device)

    with torch.no_grad():
        model.eval()
        random_init_accuracies = [ [] for _ in range(num_tasks) ]

        for batch in tqdm(test_dataloder, desc="Calculating random init accuracies"):
            images, targets, task_ids = batch[0], batch[1], batch[2]
            predictions = model(images.to(device))
            _, predicted_labels = predictions.topk(1)

            # Calculate accuracy for this batch
            correct = predicted_labels.flatten() == targets.to(device)
            for id in range(num_tasks):
                indexes = task_ids == id
                random_init_accuracies[id].extend(correct[indexes].float().cpu().numpy().tolist())
        random_init_accuracies = [np.mean(np.array(acc)) for acc in random_init_accuracies]

    return np.array(random_init_accuracies)
    