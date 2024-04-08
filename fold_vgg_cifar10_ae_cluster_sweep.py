import os
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

from folding.models.vgg import VGG
from folding.fold_cfg import BaseFoldCfg
from folding.fold import NetworkFold
from folding.folding_methods.autoencoder_clustering import ClusterFoldingConv

from torchvision.transforms.functional import rotate


def get_datasets(train=True, bs=16):
    path   = os.path.dirname(os.path.abspath(__file__))

    transform_train = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(), # FLips the image w.r.t horizontal axis
            transforms.RandomRotation(10),     #Rotates the image to a specified angel
            transforms.RandomAffine(0, shear=10, scale=(0.8,1.2)), #Performs actions like zooms, change shear angles.
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), # Set the color params
            transforms.ToTensor(), # comvert the image to tensor so that it can work with torch
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.ToTensor(), # comvert the image to tensor so that it can work with torch
        ]
    )

    transform = transform_train
    if train is False:
        transform = transform_test

    mnistTrainSet = torchvision.datasets.CIFAR10(
        root=path + '/data', 
        train=train,
        download=True, 
        transform=transform
    )

    first_half = [
        idx for idx, target in enumerate(mnistTrainSet.targets) 
        if target in [0, 1, 2, 3, 4]
    ]

    second_half = [
        idx for idx, target in enumerate(mnistTrainSet.targets) 
        if target in [5, 6, 7, 8, 9]
    ]  

    FirstHalfLoader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(mnistTrainSet, first_half),
        batch_size=bs, #256
        shuffle=True,
        num_workers=8)
    
    SecondHalfLoader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(mnistTrainSet, second_half),
        batch_size=bs, #256
        shuffle=True,
        num_workers=8)
    
    return FirstHalfLoader, SecondHalfLoader


def main():
    loader_train, _ = get_datasets(train=True)
    loader_test, _  = get_datasets(train=False)
    fold_cfg        = BaseFoldCfg(num_experiments=1)
    vgg11_cfg       = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
    
    models = dict()
    configs = dict()
    loaders = dict()
    names  = dict()


    for idx, div in enumerate((np.arange(10) + 1) ** 2 + 0.5):
        models.update({
            idx: {
                "model": VGG,
                "args": {
                    "w": 2,
                    "cfg": vgg11_cfg,
                    "classes": 10,
                    "bnorm": True
                }
            }
        })
        configs.update({
            idx: {
                "fold_method": ClusterFoldingConv(
                    loader_train,
                    epochs=2,
                    device="cuda",
                    r=0.5 #1.0 / div
                ),
                "device": "cuda"
            }
        })
        loaders.update({
            idx: {
                "loader_train": loader_train,
                "loader_test": loader_test
            }
        })
        names.update({
            idx: {
                "experiment_name": "fold_vgg_cifar10_ae_decomp_0",
                "model_name": "vgg_cifar_10_first",
            }
        })
        break
    
    fold_cfg.models = models
    fold_cfg.configs = configs
    fold_cfg.loaders = loaders
    fold_cfg.names = names
    fold_cfg.root_path = os.path.dirname(os.path.abspath(__file__))
    fold_cfg.proj_name = "fold_vgg_cifar10_ae_decomp"

    folding = NetworkFold(fold_cfg, log_wandb=False)
    accs = folding()
    divs = 1.0 / (np.arange(10) + 1) ** 2 + 0.5



if __name__ == "__main__":
    main()