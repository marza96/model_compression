import os
import torch
import torchvision
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
    fold_cfg     = BaseFoldCfg(num_experiments=1)

    fold_cfg.models = {
        0: {
            "model": MLP,
            "args": {
                "layers": 5,
                "channels": 512,
                "classes": 10,
            }
        }
    }
    fold_cfg.configs = {
        0: {
            "fold_method": NaiveAutoencoder(
                loader_train,
                epochs=1,
                device="cuda"
            ),
            "device": "cuda"
        }
    }
    fold_cfg.loaders = {
        0: {
            "loader_train": loader_train,
            "loader_test": loader_test
        }
    }
    fold_cfg.names = {
        0: {
            "experiment_name": "fold_mlp_fashion_mnist_ae_naive_0",
            "model_name": "mlp_fashion_mnist",
        }
    }
    fold_cfg.root_path = os.path.dirname(os.path.abspath(__file__))
    fold_cfg.proj_name = "fold_mlp_fashion_mnist_ae_naive"

    folding = NetworkFold(fold_cfg, log_wandb=True)
    folding()


if __name__ == "__main__":
    main()