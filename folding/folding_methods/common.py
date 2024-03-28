import torch.nn as nn
import numpy as np
import torch
import wandb
import tqdm
import scipy

from tensorly.decomposition import partial_tucker


def low_rank_approx(SVD=None, A=None, r=1):
    """
    Computes an r-rank approximation of a matrix
    given the component u, s, and v of it's SVD
    Requires: numpy
    """
    if not SVD:
        SVD = np.linalg.svd(A, full_matrices=False)
    u, s, v = SVD
    Ar = np.zeros((len(u), len(v)))
    for i in range(r):
        Ar += s[i] * np.outer(u.T[i], v[i])

    return Ar


class ConvDecompAe(nn.Module):
    def __init__(self, layer, input_dim, latent_dim, device, layer_idx=None):
        super().__init__()

        self.layer_idx = layer_idx
        ret, _ = partial_tucker(
            layer.weight.data.cpu().numpy(), 
            modes=(0, 1), 
            rank=(latent_dim, input_dim), 
            init='svd'
        )

        core        = ret[0]
        last, first = ret[1]

        self.core = torch.nn.Parameter(torch.tensor(core.copy(), device=device))
        self.last = torch.nn.Parameter(torch.tensor(last.copy(), device=device))
        self.first = torch.nn.Parameter(torch.tensor(first.copy(), device=device))
        
        self.first_layer = torch.nn.Conv2d(in_channels=first.shape[0], \
                out_channels=first.shape[1], kernel_size=1,
                stride=1, padding=0, dilation=layer.dilation, 
                bias=False).to(device)

        self.core_layer = torch.nn.Conv2d(in_channels=core.shape[1], \
                out_channels=core.shape[0], kernel_size=layer.kernel_size,
                stride=layer.stride, padding=layer.padding, dilation=layer.dilation,
                bias=False).to(device)

        self.last_layer = torch.nn.Conv2d(in_channels=last.shape[1], \
                out_channels=last.shape[0], kernel_size=1, stride=1,
                padding=0, dilation=layer.dilation, 
                bias=True).to(device)
        
        self.first_layer.weight.data = torch.transpose(self.first.detach().clone(), 1, 0).unsqueeze(-1).unsqueeze(-1)
        self.last_layer.weight.data = self.last.detach().clone().unsqueeze(-1).unsqueeze(-1)
        self.core_layer.weight.data = self.core.detach().clone()
        self.last_layer.bias.data = layer.bias.data.detach().clone()
        self.activation = torch.nn.ReLU()

    def forward(self, x):
        x = self.first_layer(x)
        x = self.core_layer(x)


        x = self.activation(x)

        x = self.last_layer(x)
    
        return x
    

class DenseDecompAE(nn.Module):
    def __init__(self, weight, latent_dim, device):
        super().__init__()

        U, S, Vh = scipy.sparse.linalg.svds(weight.detach().cpu().numpy(), k=latent_dim)
        S = np.diag(S)

        self.weight_o = nn.Parameter(torch.tensor(U.copy(), device=device))
        self.weight   = nn.Parameter(torch.tensor(S @ Vh, device=device))
        self.bnorm      = nn.BatchNorm1d(latent_dim).to(device)
        self.activation = torch.nn.ReLU()

    def forward(self, x):
        x = torch.nn.functional.linear(x, self.weight, None)
        x = self.activation(x)
        x = torch.nn.functional.linear(x, self.weight_o, None)

        return x
    

class DecompAEWrapper(nn.Module):
    def __init__(self, layer, autoencoder):
        super().__init__()

        self.layer       = layer
        self.autoencoder = autoencoder
        self.out_ae      = None
        self.out         = None

    def forward(self, x):
        out         = self.layer(x)
        self.out    = out
        
        out_ae      = self.autoencoder(x)
        self.out_ae = out_ae
        
        return out_ae
    
    def toggle_train(self):
        self.autoencoder.train()

        self.layer.weight.requires_grad = False
        self.layer.bias.requires_grad   = False

        self.layer.eval()

        

class DenseNaiveAE(nn.Module):
    def __init__(self, feature_dim, latent_dim, device):
        super().__init__()

        self.weight     = nn.Parameter(torch.rand(latent_dim, feature_dim).to(device))
        self.weight_o   = nn.Parameter(torch.rand(feature_dim, latent_dim).to(device))
        self.activation = torch.nn.ReLU()
        self.bnorm      = nn.BatchNorm1d(latent_dim).to(device)

    def forward(self, x):
        x_lat = torch.nn.functional.linear(x, self.weight, None)
        x_lat = self.activation(x_lat)
        x_lat = self.bnorm(x_lat)
        x_out = torch.nn.functional.linear(x_lat, self.weight_o, None)

        return x_out
    

class DenseAEWrapper(nn.Module):
    def __init__(self, layer, autoencoder):
        super().__init__()

        self.layer       = layer
        self.autoencoder = autoencoder
        self.out_ae      = None
        self.out         = None

    def forward(self, x):
        out         = self.layer(x)
        self.out    = out
        out         = self.autoencoder(out)
        self.out_ae = out

        return out
    
    def toggle_train(self):
        self.layer.eval()
        self.layer.weight.requires_grad = False
        self.layer.bias.requires_grad = False
        self.autoencoder.train()
        self.autoencoder.bnorm.train()
        self.autoencoder.apply(init_weights)


def init_weights(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_normal(m.weight)
        

def train_loop(model, layer_idx, optimizer, loss_fn, epochs, train_loader, device="cuda"):    
    for _ in tqdm.tqdm(range(epochs), desc="AE Tuning"):
        total        = 0
        ce_loss_acum = 0.0

        for _, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad(set_to_none=True)
            outputs = model(inputs.to(device)) 
            

            ce_loss = torch.nn.functional.cross_entropy(outputs, labels.to(device))

            out = model.layers[layer_idx].out
            out_ae = model.layers[layer_idx].out_ae
            loss   = loss_fn(out_ae, out)#+ l1_loss
            ce_loss.backward()

            ce_loss_acum += ce_loss.detach().cpu()

            total += 1
            optimizer.step()

            del loss, outputs, out, out_ae, ce_loss

        train_loss = ce_loss_acum / total

        # wandb.log({"train_loss": train_loss, "sparsity_loss": sparsity_loss})

        print(train_loss)
    