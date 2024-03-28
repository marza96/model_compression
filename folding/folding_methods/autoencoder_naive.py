import torch
import numpy as np
import matplotlib.pyplot as plt

from .base_method import BaseMethod
from .common import DenseAEWrapper, DenseNaiveAE, train_loop


class NaiveAutoencoder(BaseMethod):
    def __init__(self, loader, epochs, r=0.55, device="cuda"):
        super().__init__(loader, epochs, r=r, device=device)

    def __call__(self, model, log_wandb=True):
        model.to(self.device)
        
        i = 0
        for block in model.spec.layer_spec_unique:
            for layer in block:
                layer_idx = int(layer.split(".")[-1])
                if layer_idx == 0:
                    continue
                
                if type(model.layers[layer_idx]) is torch.nn.Linear:
                    feature_dim             = model.layers[layer_idx].weight.shape[0]
                    model.layers[layer_idx] = DenseAEWrapper(
                        model.layers[layer_idx],
                        DenseNaiveAE(feature_dim, int(self.r * feature_dim), self.device)
                    )

                    model.eval()
                    model.layers[layer_idx].toggle_train()
                    model.layers[layer_idx + 1] = torch.nn.Identity()

                    optim = torch.optim.Adam(
                        model.layers[layer_idx].autoencoder.parameters(),
                        0.005,
                    )
                    loss_fn = torch.nn.HuberLoss(delta=0.1)
                                                                #55
                    train_loop(model, layer_idx, optim, loss_fn, self.epochs, self.loader, self.device)
                    i += 1
                    # mat1 = model.layers[layer_idx].autoencoder.weight.data.cpu().numpy()
                    # mat2 = model.layers[layer_idx].layer.weight.data.cpu().numpy()
                    # mat = mat1 @ mat2

                    if i == 4:
                        break

            if i == 4:
                break

        return model.to(self.device)