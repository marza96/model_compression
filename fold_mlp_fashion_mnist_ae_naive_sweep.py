import os
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

from folding.models.mlp import MLP
from folding.fold_cfg import BaseFoldCfg
from folding.fold import NetworkFold
from folding.folding_methods.autoencoder_naive import NaiveAutoencoder

from torchvision.transforms.functional import rotate


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


def main():
    loader_train = get_datasets(train=True)
    loader_test  = get_datasets(train=False)
    fold_cfg     = BaseFoldCfg(num_experiments=10)
    
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
                "fold_method": NaiveAutoencoder(
                    loader_train,
                    epochs=2,
                    device="cuda",
                    r=1.0 / div
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
    
    fold_cfg.models = models
    fold_cfg.configs = configs
    fold_cfg.loaders = loaders
    fold_cfg.names = names

    fold_cfg.root_path = os.path.dirname(os.path.abspath(__file__))
    fold_cfg.proj_name = "fold_mlp_fashion_mnist_ae_decomp"

    folding = NetworkFold(fold_cfg, log_wandb=True)
    accs = folding()
    divs = 1.0 / (np.arange(10) + 1) ** 2 + 0.5
    plt.plot(divs, accs)
    plt.savefig("plot")


if __name__ == "__main__":
    main()