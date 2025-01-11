import os
import argparse
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np

from simclr import SimCLR
from simclr.modules import LogisticRegression, get_resnet
from simclr.modules.transformations import TransformsSimCLR
from simclr.modules.dataloader_transform import ImageMaskDataset
from simclr.modules.early_stopping import EarlyStopping
from simclr.modules.utils import check_duplicates

from model import load_optimizer

from utils import yaml_config_hook


def setup_data():
    parser = argparse.ArgumentParser(description="SimCLR")
    config = yaml_config_hook("./config/config.yaml")

    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))

    args, unknown = parser.parse_known_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    if args.dataset == "STL10":
        train_dataset = torchvision.datasets.STL10(
            args.dataset_dir,
            split="train",
            download=True,
            transform=TransformsSimCLR(size=args.image_size).test_transform,
                )
        test_dataset = torchvision.datasets.STL10(
            args.dataset_dir,
            split="test",
            download=True,
            transform=TransformsSimCLR(size=args.image_size).test_transform,
                )
    elif args.dataset == "CIFAR10":
        train_dataset = torchvision.datasets.CIFAR10(
            args.dataset_dir,
            train=True,
            download=True,
            transform=TransformsSimCLR(size=args.image_size).test_transform,
                )
        test_dataset = torchvision.datasets.CIFAR10(
            args.dataset_dir,
            train=False,
            download=True,
            transform=TransformsSimCLR(size=args.image_size).test_transform,
                )
    elif args.dataset == 'FISH':
        train_dataset = ImageMaskDataset(
        bucket_name = args.bucket_name,
        image_size=args.image_size,
        train=True,
        unlabeled=False,
        unlabeled_split_percentage=0.90,
        transform = TransformsSimCLR(size=args.image_size).test_transform)

        val_dataset = ImageMaskDataset(
        bucket_name = args.bucket_name,
        image_size=args.image_size,
        unlabeled=False,
        train=False,
        test=False,
        unlabeled_split_percentage=0.90,
        transform = TransformsSimCLR(size=args.image_size).test_transform)

        test_dataset = ImageMaskDataset(
        bucket_name = args.bucket_name,
        image_size=args.image_size,
        unlabeled=False,
        train=False,
        test=True,
        unlabeled_split_percentage=0.90,
        transform = TransformsSimCLR(size=args.image_size).test_transform)
    else:
        raise NotImplementedError
    
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.logistic_batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.workers,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.logistic_batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=args.workers,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.logistic_batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=args.workers,
    )
    
    return train_loader, val_loader, test_loader, args