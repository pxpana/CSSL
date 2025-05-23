import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
from torchvision import datasets
from torchvision.transforms import v2 as transforms
from utils.toolkit import split_images_labels
from torchvision.datasets import ImageFolder
from typing import Callable, Optional, Tuple, Union

from dataset import PretrainDataset

from continuum.datasets import CIFAR100
from continuum import ClassIncremental
from continuum.generators import ClassOrderGenerator


class iData(object):
    train_trsf = []
    test_trsf = []
    common_trsf = []
    class_order = None


class iCIFAR100(iData):
    use_path = True
    train_trsf = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=63 / 255),
    ]
    test_trsf = []
    common_trsf = [
        transforms.Compose([transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True)]),
        transforms.Normalize(
            mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761)
        ),
    ]

    class_order = np.arange(100).tolist()

    def download_data(self):
        train_dataset = ImageFolder(f"data/CIFAR100/train")
        test_dataset = ImageFolder(f"data/CIFAR100/test")
        self.train_data, self.train_targets = split_images_labels(train_dataset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dataset.imgs)

class DataManager():
    def __init__(self, args, root="data"):

        self.num_tasks = args.num_tasks

        train_dataset = CIFAR100(root, download=True, train=True)
        self.test_dataset = CIFAR100(root, download=True, train=False)
        train_classifier_transform, test_classifier_transform = prepare_transforms("cifar100")

        train_pretrain_scenario = ClassIncremental(
            train_dataset,
            nb_tasks=self.num_tasks,
            transformations=[None]
        )

        train_classifier_scenario = ClassIncremental(
            train_dataset,
            nb_tasks=self.num_tasks,
            transformations=[train_classifier_transform]
        )

        self.test_classifier_transform = test_classifier_transform

        self.train_pretrain_generator = ClassOrderGenerator(train_pretrain_scenario)
        self.train_classifier_generator = ClassOrderGenerator(train_classifier_scenario)

    def get_scenerio(self, scenario_id: int):
        train_pretrain_scenario = self.train_pretrain_generator.sample(seed=scenario_id)
        train_classifier_scenario = self.train_classifier_generator.sample(seed=scenario_id)

        test_classifier_scenario = ClassIncremental(
            self.test_dataset,
            nb_tasks=self.num_tasks,
            transformations=[self.test_classifier_transform],
            class_order=train_classifier_scenario.class_order
        )

        return train_pretrain_scenario, train_classifier_scenario, test_classifier_scenario
    
    def get_dataloader(
            self, 
            train_pretrain_taskset, 
            train_classifier_taskset, 
            test_classifier_taskset, 
            pretrain_transform,
            args
        ):

        pretrain_dataset = PretrainDataset(
            data=train_pretrain_taskset._x,
            transform=pretrain_transform,
        )

        train_pretrain_loader = DataLoader(
            pretrain_dataset, 
            batch_size=args.train_batch_size, 
            num_workers=args.num_workers, 
            shuffle=True
        )

        train_classifier_loader = DataLoader(train_classifier_taskset, batch_size=args.test_batch_size, num_workers=args.num_workers, shuffle=True)
        test_classifier_loader = DataLoader(test_classifier_taskset, batch_size=args.test_batch_size, num_workers=args.num_workers, shuffle=False)

        return train_pretrain_loader, train_classifier_loader, test_classifier_loader


        


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

    #custom_pipeline = build_custom_pipeline()

    pipelines = {
        "cifar10": cifar_pipeline,
        "cifar100": cifar_pipeline,
        "stl10": stl_pipeline,
        "imagenet100": imagenet_pipeline,
        "imagenet": imagenet_pipeline,
        "domainnet": imagenet_pipeline,
        #"custom": custom_pipeline,
    }

    assert dataset in pipelines

    pipeline = pipelines[dataset]
    T_train = pipeline["T_train"]
    T_val = pipeline["T_val"]

    return T_train, T_val
