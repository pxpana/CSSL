import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
from torchvision import datasets
from torchvision.transforms import v2 as transforms
from cssl.utils.toolkit import split_images_labels
from torchvision.datasets import ImageFolder
from typing import Callable, Optional, Tuple, Union

from cssl.dataset import PretrainDataset

from continuum.datasets import CIFAR100, MNIST
from continuum import ClassIncremental
from continuum.generators import ClassOrderGenerator


class DataManager():
    def __init__(self, args, root="data"):

        self.num_tasks = args.num_tasks

        train_dataset, self.test_dataset = get_dataset(dataset=args.dataset, root=root)
        train_classifier_transform, test_classifier_transform = prepare_transforms(dataset=args.dataset)

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
            transform=pretrain_transform
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


        
def get_dataset(dataset, root):
    if dataset == "cifar100":
        train_dataset = CIFAR100(root, download=True, train=True)
        test_dataset = CIFAR100(root, download=True, train=False)
    elif dataset == "mnist":
        train_dataset = MNIST(root, download=True, train=True)
        test_dataset = MNIST(root, download=True, train=False)

    return train_dataset, test_dataset

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
