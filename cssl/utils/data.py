import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
from torchvision import datasets
from torchvision.transforms import v2 as transforms
from cssl.utils.toolkit import split_images_labels
from torchvision.datasets import ImageFolder
from typing import Any, Callable, Iterable, List, Optional, Sequence, Type, Union, Tuple

from cssl.dataset import PretrainDataset

#from continuum.datasets import CIFAR100, MNIST
from torchvision.datasets import CIFAR100
from continuum import ClassIncremental
from continuum.generators import ClassOrderGenerator

from torch.utils.data.dataset import Dataset, Subset

class DataManager():
    def __init__(self, args, root="data"):

        self.num_tasks = args.num_tasks
        self.tasks = None
        if args.split_strategy == "class":
            assert args.num_classes % args.num_tasks == 0
            self.tasks = torch.randperm(args.num_classes).chunk(args.num_tasks)

        self.train_dataset, self.test_dataset = get_dataset(dataset=args.dataset, root=root)
        self.train_classifier_transform, test_classifier_transform = prepare_transforms(dataset=args.dataset)

        self.train_pretrain_scenario = []
        for task_idx in range(self.num_tasks):
            task_dataset, _ = split_dataset(
                dataset=self.train_dataset,
                task_idx=task_idx,
                num_tasks=self.num_tasks,
                split_strategy=args.split_strategy,
                tasks=self.tasks
            )
            self.train_pretrain_scenario.append(task_dataset)
        self.pretrain_transform = get_pretrain_transform(args)

        self.test_classifier_transform = test_classifier_transform


    


        
def get_dataset(dataset, root):
    if dataset == "cifar100":
        train_dataset = CIFAR100(root, download=True, train=True)
        test_dataset = CIFAR100(root, download=True, train=False)
    elif dataset == "mnist":
        train_dataset = MNIST(root, download=True, train=True)
        test_dataset = MNIST(root, download=True, train=False)

    return train_dataset, test_dataset

def split_dataset(
    dataset: Dataset, task_idx: List[int], num_tasks: int, split_strategy: str, tasks: list = None
):
    if split_strategy == "class":
        assert len(dataset.classes) == sum([len(t) for t in tasks])
        mask = [(c in tasks[task_idx]) for c in dataset.targets]
        indexes = torch.tensor(mask).nonzero()
        task_dataset = Subset(dataset, indexes)
    elif split_strategy == "data":
        assert tasks is None
        lengths = [len(dataset) // num_tasks] * num_tasks
        lengths[0] += len(dataset) - sum(lengths)
        task_dataset = torch.utils.data.random_split(
            dataset, lengths, generator=torch.Generator().manual_seed(42)
        )[task_idx]
    elif split_strategy == "domain":
        assert tasks is None
        raise NotImplementedError
    return task_dataset, tasks

def prepare_transforms(dataset: str) -> Tuple[nn.Module, nn.Module]:
    """
    Prepares pre-defined train and test transformation pipelines for some datasets.

    Implementation from CaSSLE [0]

    - [1] https://github.com/DonkeyShot21/cassle/blob/main/cassle/utils/classification_dataloader.py

    Args:
        dataset (str): dataset name.

    Returns:
        Tuple[nn.Module, nn.Module]: training and validation transformation pipelines.
    """

    cifar_pipeline = {
        "T_train": transforms.Compose(
            [
                transforms.RandomResizedCrop(size=32, scale=(0.08, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.Compose([transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True)]),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
            ]
        ),
        "T_val": transforms.Compose(
            [
                transforms.Compose([transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True)]),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
            ]
        ),
    }

    stl_pipeline = {
        "T_train": transforms.Compose(
            [
                transforms.RandomResizedCrop(size=96, scale=(0.08, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.Compose([transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True)]),
                transforms.Normalize((0.4914, 0.4823, 0.4466), (0.247, 0.243, 0.261)),
            ]
        ),
        "T_val": transforms.Compose(
            [
                transforms.Resize((96, 96)),
                transforms.Compose([transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True)]),
                transforms.Normalize((0.4914, 0.4823, 0.4466), (0.247, 0.243, 0.261)),
            ]
        ),
    }

    imagenet_pipeline = {
        "T_train": transforms.Compose(
            [
                transforms.RandomResizedCrop(size=224, scale=(0.08, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.Compose([transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True)]),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.228, 0.224, 0.225)),
            ]
        ),
        "T_val": transforms.Compose(
            [
                transforms.Resize(256),  # resize shorter
                transforms.CenterCrop(224),  # take center crop
                transforms.Compose([transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True)]),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.228, 0.224, 0.225)),
            ]
        ),
    }

    mnist_pipeline = {
        "T_train": transforms.Compose(
            [
                transforms.RandomResizedCrop(size=28, scale=(0.08, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.Grayscale(num_output_channels=1),
                transforms.ToImage(),  # Only in TorchVision >= 0.15
                transforms.ToDtype(torch.float32, scale=True),  # Scale to [0,1]
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        ),
        "T_val": transforms.Compose(
            [
                transforms.Resize((28, 28)),
                transforms.Grayscale(num_output_channels=1),
                transforms.ToImage(),  # Only in TorchVision >= 0.15
                transforms.ToDtype(torch.float32, scale=True),  # Scale to [0,1]
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        ),
    }

    #custom_pipeline = build_custom_pipeline()

    pipelines = {
        "cifar10": cifar_pipeline,
        "cifar100": cifar_pipeline,
        "stl10": stl_pipeline,
        "imagenet100": imagenet_pipeline,
        "imagenet": imagenet_pipeline,
        "domainnet": imagenet_pipeline,
        "mnist": mnist_pipeline,
        #"custom": custom_pipeline,
    }

    assert dataset in pipelines

    pipeline = pipelines[dataset]
    T_train = pipeline["T_train"]
    T_val = pipeline["T_val"]

    return T_train, T_val

def get_pretrain_transform(args):
    dataset_name = args.dataset.lower()
    name = args.model_name.lower()
    pretrain_collate_function=None

    if dataset_name in ["cifar100"]:
        from cssl.utils import CifarTransform
        dataset_transform1 = CifarTransform(
            brightness=args.brightness,
            contrast=args.contrast,
            saturation=args.saturation,
            hue=args.hue,
            gaussian_prob=args.gaussian_blur[0],
            solarization_prob=args.solarization[0],
        )

        dataset_transform2 = CifarTransform(
            brightness=args.brightness,
            contrast=args.contrast,
            saturation=args.saturation,
            hue=args.hue,
            gaussian_prob=args.gaussian_blur[1],
            solarization_prob=args.solarization[1],
        )

    if name in ["simclr", "byol", "barlowtwins", "mocov2plus"]:
        from lightly.transforms.multi_view_transform import MultiViewTransform
        transform = MultiViewTransform([dataset_transform1, dataset_transform2])
    elif name == "simsiam":
        from lightly.transforms import SimSiamTransform
        transform = SimSiamTransform(input_size=args.image_dim)
    elif name == "vicreg":
        from lightly.transforms import VICRegTransform
        transform = VICRegTransform(input_size=args.image_dim)
    elif name == "swav":
        from lightly.transforms import SwaVTransform
        transform = SwaVTransform(crop_sizes=args.crop_sizes)

    else:
        assert 0
    return transform

import random
from PIL import Image, ImageFilter, ImageOps
from typing import Any, Callable, Iterable, List, Optional, Sequence, Type, Union

class BaseTransform:
    """Adds callable base class to implement different transformation pipelines."""

    def __call__(self, x: Image) -> torch.Tensor:
        return self.transform(x)

    def __repr__(self) -> str:
        return str(self.transform)
    
class CifarTransform(BaseTransform):
    def __init__(
        self,
        brightness: float,
        contrast: float,
        saturation: float,
        hue: float,
        gaussian_prob: float = 0.0,
        solarization_prob: float = 0.0,
        min_scale: float = 0.08,
    ):
        """Applies cifar transformations.

        Args:
            brightness (float): sampled uniformly in [max(0, 1 - brightness), 1 + brightness].
            contrast (float): sampled uniformly in [max(0, 1 - contrast), 1 + contrast].
            saturation (float): sampled uniformly in [max(0, 1 - saturation), 1 + saturation].
            hue (float): sampled uniformly in [-hue, hue].
            gaussian_prob (float, optional): probability of applying gaussian blur. Defaults to 0.0.
            solarization_prob (float, optional): probability of applying solarization. Defaults
                to 0.0.
            min_scale (float, optional): minimum scale of the crops. Defaults to 0.08.
        """

        super().__init__()

        self.transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    (32, 32),
                    scale=(min_scale, 1.0),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness, contrast, saturation, hue)], p=0.8
                ),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([GaussianBlur()], p=gaussian_prob),
                transforms.RandomApply([Solarization()], p=solarization_prob),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
            ]
        )

class GaussianBlur:
    def __init__(self, sigma: Sequence[float] = [0.1, 2.0]):
        """Gaussian blur as a callable object.

        Args:
            sigma (Sequence[float]): range to sample the radius of the gaussian blur filter.
                Defaults to [0.1, 2.0].
        """

        self.sigma = sigma

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Applies gaussian blur to an input image.

        Args:
            x (torch.Tensor): an image in the tensor format.

        Returns:
            torch.Tensor: returns a blurred image.
        """

        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class Solarization:
    """Solarization as a callable object."""

    def __call__(self, img: Image) -> Image:
        """Applies solarization to an input image.

        Args:
            img (Image): an image in the PIL.Image format.

        Returns:
            Image: a solarized image.
        """

        return ImageOps.solarize(img)
