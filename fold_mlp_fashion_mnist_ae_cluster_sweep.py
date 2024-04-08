import os
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

from folding.models.mlp import MLP
from folding.fold_cfg import BaseFoldCfg
from folding.fold import NetworkFold
from folding.folding_methods.autoencoder_clustering import ClusterFolding


def get_datasets(train=True):
    path   = os.path.dirname(os.path.abspath(__file__))

    transform_train = transforms.Compose(
        [
            # transforms.RandomHorizontalFlip(), # FLips the image w.r.t horizontal axis
            # transforms.RandomRotation(10),     #Rotates the image to a specified angel
            # transforms.RandomAffine(0, shear=10, scale=(0.8,1.2)), #Performs actions like zooms, change shear angles.
            # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), # Set the color params
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
        
    dataset = torchvision.datasets.FashionMNIST(
        root=path + '/data', 
        train=train,
        download=True, 
        transform=transform
    )   

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,
        num_workers=8)
    
    return loader


def main():
    loader_train = get_datasets(train=True)
    loader_test  = get_datasets(train=False)
    fold_cfg     = BaseFoldCfg(num_experiments=1)
    
    models = dict()
    configs = dict()
    loaders = dict()
    names  = dict()


    for idx, div in enumerate((np.arange(10) + 1) ** 2 + 0.5):
        models.update({
            idx: {
                "model": MLP,
                "args": {
                    "layers": 5,
                    "channels": 512,
                    "classes": 10,
                }
            }
        })
        configs.update({
            idx: {
                "fold_method": ClusterFolding(
                    loader_train,
                    epochs=1,
                    device="cuda",
                    r=0.15
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
                "experiment_name": "fold_mlp_fashion_mnist_ae_decomp_0",
                "model_name": "mlp_fashion_mnist",
            }
        })
        break
    
    fold_cfg.models = models
    fold_cfg.configs = configs
    fold_cfg.loaders = loaders
    fold_cfg.names = names

    fold_cfg.root_path = os.path.dirname(os.path.abspath(__file__))
    fold_cfg.proj_name = "fold_mlp_fashion_mnist_ae_decomp"

    folding = NetworkFold(fold_cfg, log_wandb=False)
    accs = folding()
    divs = 1.0 / (np.arange(10) + 1) ** 2 + 0.5


if __name__ == "__main__":
    main()