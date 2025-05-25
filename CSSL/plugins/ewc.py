from collections import defaultdict
from typing import Dict, Tuple, Union
import warnings
import itertools

import torch
from torch.utils.data import DataLoader

from avalanche.models.utils import avalanche_forward
from avalanche.training.plugins.strategy_plugin import SupervisedPlugin
from avalanche.training.utils import copy_params_dict, zerolike_params_dict, ParamData


class EWC:
    def __init__(
        self,
        task_id,
        config
    ):
        super().__init__()

        self.ewc_lambda = config.ewc_lambda
        self.decay_factor = config.ewc_decay_factor
        self.num_workers = config.num_workers
        self.task_id = task_id


        self.saved_params: Dict[int, Dict[str, ParamData]] = defaultdict(dict)
        self.importances: Dict[int, Dict[str, ParamData]] = defaultdict(dict)

    def on_before_backward(self, trainer, pl_module, loss):
        """
        Compute EWC penalty and add it to the loss.
        """
        if self.task_id == 0:
            return

        penalty = torch.tensor(0).float().to(self.device)

        for experience in range(self.task_id):
            for k, cur_param in self.backbone.named_parameters():
                # new parameters do not count
                if k not in self.saved_params[experience]:
                    continue
                saved_param = self.saved_params[experience][k]
                imp = self.importances[experience][k]
                new_shape = cur_param.shape
                penalty += (
                    imp.expand(new_shape)
                    * (cur_param - saved_param.expand(new_shape)).pow(2)
                ).sum()

        loss += self.ewc_lambda * penalty

    def on_train_end(self, trainer, pl_module):
        """
        Compute importances of parameters after each experience.
        """
        importances = self.compute_importances(
            self.backbone,
            self._criterion,
            self.optimizer,
            strategy.experience.dataset,
            self.device,
            strategy.train_mb_size,
            num_workers=self.num_workers),
        )
        self.update_importances(importances, self.task_id)
        self.saved_params[self.task_id] = copy_params_dict(self.backbone)
        # clear previous parameter values
        del self.saved_params[self.task_id - 1]

    def compute_importances(
        self, model, criterion, optimizer, dataloader) -> Dict[str, ParamData]:
        """
        Compute EWC importance matrix for each parameter
        """

        model.eval()

        # Set RNN-like modules on GPU to training mode to avoid CUDA error
        if device == "cuda":
            for module in model.modules():
                if isinstance(module, torch.nn.RNNBase):
                    warnings.warn(
                        "RNN-like modules do not support "
                        "backward calls while in `eval` mode on CUDA "
                        "devices. Setting all `RNNBase` modules to "
                        "`train` mode. May produce inconsistent "
                        "output if such modules have `dropout` > 0."
                    )
                    module.train()

        # list of list
        importances = zerolike_params_dict(model)
        for i, batch in enumerate(dataloader):
            # get only input, target and task_id from the batch
            x, y, task_labels = batch[0], batch[1], batch[-1]
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            out = avalanche_forward(model, x, task_labels)
            loss = criterion(out, y)
            loss.backward()

            for (k1, p), (k2, imp) in zip(
                model.named_parameters(), importances.items()
            ):
                assert k1 == k2
                if p.grad is not None:
                    imp.data += p.grad.data.clone().pow(2)

        # average over mini batch length
        for _, imp in importances.items():
            imp.data /= float(len(dataloader))

        model.train()

        return importances

    @torch.no_grad()
    def update_importances(self, importances, t: int):
        """
        Update importance for each parameter based on the currently computed
        importances.
        """

        if self.mode == "separate" or t == 0:
            self.importances[t] = importances
        elif self.mode == "online":
            for (k1, old_imp), (k2, curr_imp) in itertools.zip_longest(
                self.importances[t - 1].items(),
                importances.items(),
                fillvalue=(None, None),
            ):
                # Add new module importances to the importances value (New head)
                if k1 is None:
                    assert k2 is not None
                    assert curr_imp is not None
                    self.importances[t][k2] = curr_imp
                    continue

                assert k1 == k2, "Error in importance computation."
                assert curr_imp is not None
                assert old_imp is not None
                assert k2 is not None

                # manage expansion of existing layers
                self.importances[t][k1] = ParamData(
                    f"imp_{k1}",
                    curr_imp.shape,
                    init_tensor=self.decay_factor * old_imp.expand(curr_imp.shape)
                    + curr_imp.data,
                    device=curr_imp.device,
                )

            # clear previous parameter importances
            if t > 0 and (not self.keep_importance_data):
                del self.importances[t - 1]

        else:
            raise ValueError("Wrong EWC mode.")


ParamDict = Dict[str, Union[ParamData]]
EwcDataType = Tuple[ParamDict, ParamDict]