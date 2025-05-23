from typing import Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor


class BarlowTwinsLoss(torch.nn.Module):
    """Implementation of the Barlow Twins Loss from Barlow Twins[0] paper.

    This code specifically implements the Figure Algorithm 1 from [0].
    [0] Zbontar,J. et.al, 2021, Barlow Twins... https://arxiv.org/abs/2103.03230

    Examples:
        >>> # initialize loss function
        >>> loss_fn = BarlowTwinsLoss()
        >>>
        >>> # generate two random transforms of images
        >>> t0 = transforms(images)
        >>> t1 = transforms(images)
        >>>
        >>> # feed through SimSiam model
        >>> out0, out1 = model(t0, t1)
        >>>
        >>> # calculate loss
        >>> loss = loss_fn(out0, out1)
    """

    def __init__(self, lambda_param: float = 5e-3, scale_loss: float = 0.025, gather_distributed: bool = False):
        """Lambda param configuration with default value like in [0]

        Initializes the BarlowTwinsLoss with the specified parameters.

        Args:
            lambda_param:
                Parameter for importance of redundancy reduction term.
            gather_distributed:
                If True, the cross-correlation matrices from all GPUs are
                gathered and summed before the loss calculation.

        Raises:
            ValueError: If gather_distributed is True but torch.distributed is not available.
        """
        super(BarlowTwinsLoss, self).__init__()
        self.lambda_param = lambda_param
        self.scale_loss = scale_loss
        self.gather_distributed = gather_distributed

        if gather_distributed and not dist.is_available():
            raise ValueError(
                "gather_distributed is True but torch.distributed is not available."
                "Please set gather_distributed=False or install a torch version with "
                "distributed support."
            )

    def forward(self, z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
        """Computes the Barlow Twins loss for the given projections.

        Args:
            z_a: Output projections of the first set of transformed images.
            z_b: Output projections of the second set of transformed images.

        Returns:
            Computed Barlow Twins Loss.
        """

        # Normalize repr. along the batch dimension
        z_a_norm, z_b_norm = _normalize(z_a), _normalize(z_b)

        N = z_a.size(0)

        # Compute the cross-correlation matrix
        c = z_a_norm.T @ z_b_norm
        c.div_(N)

        # Aggregate and normalize the cross-correlation matrix between multiple GPUs
        if self.gather_distributed and dist.is_initialized():
            world_size = dist.get_world_size()
            if world_size > 1:
                c = c / world_size
                dist.all_reduce(c)

        invariance_loss = torch.diagonal(c).add_(-1).pow_(2).sum()
        redundancy_reduction_loss = _off_diagonal(c).pow_(2).sum()
        loss: Tensor = invariance_loss + self.lambda_param * redundancy_reduction_loss

        loss = self.scale_loss * loss.sum()

        return loss


def _normalize(z: torch.Tensor) -> torch.Tensor:
    """Helper function to create batches of mean 0 and std 1."""
    return F.batch_norm(
        z,
        running_mean=None,
        running_var=None,
        weight=None,
        bias=None,
        training=True,
    )


def _off_diagonal(x: Tensor) -> Tensor:
    """Returns a flattened view of the off-diagonal elements of a square matrix."""

    # Ensure the input is a square matrix
    n, m = x.shape
    assert n == m

    # Flatten the matrix and extract off-diagonal elements
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()
