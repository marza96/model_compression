import os
import copy
import torch
import torchvision

import numpy as np
import torchvision.transforms as transforms

from torch.nn import CrossEntropyLoss


from folding.train import train_from_cfg
from folding.models.vgg import VGG
from folding.train_cfg import BaseTrainCfg


def get_datasets(train=True):
    path   = os.path.dirname(os.path.abspath(__file__))

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )
    dataset = torchvision.datasets.CIFAR10(
        root=path + '/data', 
        train=train,
        download=True, 
        transform=transform
    )   

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=128,
        shuffle=True,
        num_workers=8)
    
    return loader


if __name__ == "__main__":
    loader      = get_datasets()
    loader_test = get_datasets(train=False)
    train_cfg   = BaseTrainCfg(num_experiments=1)
    vgg11_cfg   = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']

    train_cfg.proj_name = "train_vgg_cifar10"
    train_cfg.models = {
        0: {
            "model": VGG,
            "args": {
                "w": 1,
                "cfg": vgg11_cfg,
                "classes": 10,
                "bnorm": True
            }
        }
    }
    train_cfg.configs = {
        0: {
            "loss_fn": CrossEntropyLoss(),
            "epochs" : 100,
            "device": "cuda",
            "optimizer": {
                "class": torch.optim.Adam,
                "args": {
                    "lr": 0.001,
                    "weight_decay": 0.0005
                }
            }
        }
    }

    train_cfg.loaders = {
        0: {"train": loader, "test": loader_test},
    }
    train_cfg.names = {
        0: "vgg_cifar10",
    }
    train_cfg.root_path = os.path.dirname(os.path.abspath(__file__))

    train_from_cfg(train_cfg)