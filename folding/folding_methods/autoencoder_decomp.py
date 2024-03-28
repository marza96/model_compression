import torch
import numpy as np
import matplotlib.pyplot as plt

from .base_method import BaseMethod
from .common import DecompAEWrapper, DenseDecompAE, ConvDecompAe, train_loop


class DecompAutoencoder(BaseMethod):
    def __init__(self, loader, epochs, r=0.085, device="cuda"):
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
                    model.layers[layer_idx] = DecompAEWrapper(
                        model.layers[layer_idx],
                        DenseDecompAE(model.layers[layer_idx].weight, int(self.r * feature_dim), self.device)
                    )

                    model.eval()
                    model.layers[layer_idx].toggle_train()
                    model.layers[layer_idx + 1] = torch.nn.Identity()

                    optim = torch.optim.Adam(
                        model.layers[layer_idx].autoencoder.parameters(),
                        0.005,
                    )
                    loss_fn = torch.nn.HuberLoss(delta=0.1)              #55
                    train_loop(model, layer_idx, optim, loss_fn, self.epochs, self.loader, self.device)

                    i += 1
                    if i == 4:
                        break

            if i == 4:
                break

        return model.to(self.device)
    

class ConvDecompAutoencoder(BaseMethod):
    def __init__(self, loader, epochs, r=0.085, device="cuda"):
        super().__init__(loader, epochs, r=r, device=device)

    def fuse_conv_and_bn(self, conv, bn):
        fusedconv = torch.nn.Conv2d(
            conv.in_channels,
            conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            bias=True
        )

        w_conv = conv.weight.clone().view(conv.out_channels, -1).detach()
        w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps+bn.running_var))).detach()

        w_bn.requires_grad = False
        w_conv.requires_grad = False

        ww = torch.mm(w_bn.detach(), w_conv.detach())
        ww.requires_grad = False
        fusedconv.weight.data = ww.data.view(fusedconv.weight.detach().size()).detach() 
        #
        # prepare spatial bias
        if conv.bias is not None:
            b_conv = conv.bias.detach()
        else:
            b_conv = torch.zeros( conv.weight.size(0) )

        bn.bias.requires_grad = False
        bn.weight.requires_grad = False

        b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
        
        bb = ( torch.matmul(w_bn, b_conv) + b_bn ).detach()
        fusedconv.bias.data = bb.data

        return fusedconv

    def __call__(self, model, log_wandb=True):
        model.to(self.device)
        
        for block in model.spec.layer_spec_unique:
            for layer in block:
                layer_idx = int(layer.split(".")[-1])
                if layer_idx == 0:
                    continue
                
                if type(model.layers[layer_idx]) is torch.nn.Conv2d:
                    model.layers[layer_idx].eval()
                    model.layers[layer_idx + 1].eval()

                    model.layers[layer_idx] = self.fuse_conv_and_bn(
                        model.layers[layer_idx],
                        model.layers[layer_idx + 1]
                    )
                    model.layers[layer_idx + 1] = torch.nn.Identity()

        i = 0
        for block in model.spec.layer_spec_unique:
            for layer in block:
                layer_idx = int(layer.split(".")[-1])
                if layer_idx == 0:
                    continue
                
                if type(model.layers[layer_idx]) is torch.nn.Conv2d:
                    feature_dim  = model.layers[layer_idx].weight.shape[0]
                    
                    print(layer, type(model.layers[layer_idx]))
                    model.layers[layer_idx] = DecompAEWrapper(
                        model.layers[layer_idx],
                        ConvDecompAe(model.layers[layer_idx], feature_dim, int(self.r * feature_dim), self.device, layer_idx)
                    )

                    model.eval()
                    model.layers[layer_idx].toggle_train()
                    model.layers[layer_idx + 2] = torch.nn.Identity()

                    optim = torch.optim.Adam(
                        model.layers[layer_idx].autoencoder.parameters(),
                        0.0005,
                    )
                    loss_fn = torch.nn.HuberLoss(delta=0.001)              #55
                    train_loop(model, layer_idx, optim, loss_fn, self.epochs, self.loader, self.device)
                
                i += 1

                print(i)
                if i >= 4: #15
                    break
            if i >= 4: #15
                break

        return model.to(self.device)