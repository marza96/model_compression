import os
import copy
import torch
import torchvision

import numpy as np
import torchvision.transforms as transforms

from torch.optim import SGD
from torch.nn import CrossEntropyLoss
from torchvision.transforms.functional import rotate


from folding.train import train_from_cfg
from folding.models.mlp import MLP
from folding.train_cfg import BaseTrainCfg


def get_datasets(train=True):
    path   = os.path.dirname(os.path.abspath(__file__))

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )
    dataset = torchvision.datasets.FashionMNIST(
        root=path + '/data', 
        train=train,
        download=True, 
        transform=transform
    )   

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=512,
        shuffle=True,
        num_workers=8)
    
    return loader


if __name__ == "__main__":
    loader      = get_datasets()
    loader_test = get_datasets(train=False)
    train_cfg   = BaseTrainCfg(num_experiments=1)

    train_cfg.proj_name = "folding_mlp_fashion_mnist"
    train_cfg.models = {
        0: {
            "model": MLP,
            "args": {
                "layers": 5,
                "channels": 512,
                "classes": 10,
            }
        }
    }
    train_cfg.configs = {
        0: {
            "loss_fn": CrossEntropyLoss(),
            "epochs" : 35,
            "device": "cuda",
            "optimizer": {
                "class": torch.optim.Adam,
                "args": {
                    "lr": 0.0001,
                    "weight_decay": 0.005
                }
            }
        }
    }

    train_cfg.loaders = {
        0: {"train": loader, "test": loader_test},
    }
    train_cfg.names = {
        0: "mlp_fashion_mnist",
    }
    train_cfg.root_path = os.path.dirname(os.path.abspath(__file__))

    train_from_cfg(train_cfg)