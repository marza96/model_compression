import torch
import numpy as np
import tqdm

from .base_method import BaseMethod
from .common import DecompAEWrapper, train_loop
from sklearn.cluster import AgglomerativeClustering, KMeans, MiniBatchKMeans


class ClusteringAE(torch.nn.Module):
    def __init__(self, weight, bias, merge, unmerge, device):
        super().__init__()

        self.weight = torch.nn.Parameter(weight).to(device)
        self.bias   = torch.nn.Parameter(bias).to(device)

        self.merge      = torch.nn.Parameter(merge).to(device)
        self.unmerge    = torch.nn.Parameter(unmerge).to(device)
        self.activation = torch.nn.ReLU()

    def forward(self, x):
        x = torch.nn.functional.linear(x, self.weight, self.bias)
        x = torch.nn.functional.linear(x, self.merge.T, None)
        x = self.activation(x)
        x = torch.nn.functional.linear(x, self.unmerge, None)

        return x
    
class ClusteringAEConv(torch.nn.Module):
    def __init__(self, weight, bias, merge, unmerge, device):
        super().__init__()

        self.weight = torch.nn.Parameter(weight).to(device)
        self.bias   = torch.nn.Parameter(bias).to(device)

        self.merge      = torch.nn.Parameter(merge.unsqueeze(-1).unsqueeze(-1)).to(device)
        self.unmerge    = torch.nn.Parameter(unmerge.unsqueeze(-1).unsqueeze(-1)).to(device)
        self.activation = torch.nn.ReLU()

    def forward(self, x):
        x = torch.nn.functional.conv2d(x, self.weight, self.bias, padding=1)
        x = torch.nn.functional.conv2d(x, self.merge.permute(1, 0, 2, 3), None)
        x = self.activation(x)
        x = torch.nn.functional.conv2d(x, self.unmerge, None)

        return x

class CovarianceMetric():
    def __init__(self, loader, epochs, layer_idx, device="cuda"):
        self.loader    = loader
        self.epochs    = epochs
        self.layer_idx = layer_idx
        self.device    = device

    def __call__(self, net):
        net.eval()

        n      = self.epochs * len(self.loader)
        mean   = None
        std    = None
        outer  = None

        subnet = net.subnet(net, self.layer_idx)

        with torch.no_grad():
            for _ in range(self.epochs):
                for i, (images, _) in enumerate(tqdm.tqdm(self.loader, desc="corr eval")):
                    img_t = images.float().to(self.device)

                    out = subnet(img_t)
                    out = out.reshape(out.shape[0], out.shape[1], -1).permute(0, 2, 1)
                    out = out.reshape(-1, out.shape[2]).float()

                    mean_b = out.mean(dim=0)
                    std_b = out.std(dim=0)
                    outer_b = (out.T @ out) / out.shape[0]

                    if i == 0:
                        mean = torch.zeros_like(mean_b)
                        std = torch.zeros_like(std_b)
                        outer = torch.zeros_like(outer_b)

                    mean += mean_b / n
                    std += std_b / n
                    outer += outer_b / n

        cov = outer - torch.outer(mean, mean)
        corr = cov / (torch.outer(std, std) + 1e-4)

        return corr
    

class ClusterFoldingConv(BaseMethod):
    def __init__(self, loader, epochs, r=0.5, device="cuda"):
        super().__init__(loader, epochs, r, device)

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

        
        layer_idx = 4
        subnet    = model.subnet(model, layer_idx)
        feature_dim = model.layers[layer_idx].weight.shape[0]
        reduced_dim = int(self.r * feature_dim)
        kmeans = MiniBatchKMeans(n_clusters=reduced_dim,
                                 batch_size=256,
                                 n_init="auto",
                                 compute_labels=True,
                                 max_iter=10)

        with torch.no_grad():
            for _ in range(self.epochs):
                for i, (images, _) in enumerate(tqdm.tqdm(self.loader, desc="corr eval")):
                    img_t = images.float().to(self.device)
                    
                    out = subnet(img_t)
                    print(out.shape)
                    out = out.reshape(out.shape[0], out.shape[1], -1)
                    print(out.shape)
                    out = out.permute(0, 2, 1).reshape(-1, out.shape[1])
                    out = out.cpu().numpy()
                    print(out.shape)
                    if out.shape[0] < 16:
                        break
                
                    kmeans.partial_fit(out.T)
                    break
                break

        cluster_labels = kmeans.labels_
        matches = torch.zeros((feature_dim, int(self.r * feature_dim)), device=self.device)
        for model_idx, match_idx in enumerate(cluster_labels):
            matches[model_idx, match_idx] = 1
        
        merge = matches / (matches.sum(dim=0, keepdim=True) + 1e-5)
        unmerge = matches.detach().clone()

        model.layers[layer_idx] = DecompAEWrapper(
            model.layers[layer_idx],
            ClusteringAEConv(model.layers[layer_idx].weight, model.layers[layer_idx].bias, merge, unmerge, self.device)
        )
        model.layers[layer_idx + 2] = torch.nn.Identity() 

        # model.layers[layer_idx].autoencoder.weight.requires_grad = False
        # model.layers[layer_idx].autoencoder.bias.requires_grad = False
        # model.layers[layer_idx].autoencoder.merge.requires_grad = False

        # optim = torch.optim.Adam(
        #     model.layers[layer_idx].autoencoder.parameters(),
        #     0.005,
        # )
        # loss_fn = torch.nn.MSELoss()             #55
        # train_loop(model, layer_idx, optim, loss_fn, self.epochs, self.loader, self.device)

        return model.to(self.device)
    

class ClusterFolding(BaseMethod):
    def __init__(self, loader, epochs, r=0.5, device="cuda"):
        super().__init__(loader, epochs, r, device)

    def __call__(self, model, log_wandb=True):
        model.to(self.device)
        
        layer_idx = 3
        subnet    = model.subnet(model, layer_idx)
        feature_dim = model.layers[layer_idx].weight.shape[0]
        reduced_dim = int(self.r * feature_dim)
        kmeans = MiniBatchKMeans(n_clusters=reduced_dim,
                                 batch_size=512,
                                 n_init="auto",
                                 compute_labels=True,
                                 max_iter=10)

        with torch.no_grad():
            for _ in range(self.epochs):
                for i, (images, _) in enumerate(tqdm.tqdm(self.loader, desc="corr eval")):
                    img_t = images.float().to(self.device)
                    
                    out = subnet(img_t).cpu().numpy()
                    print("DBG", out.shape)
                    if out.shape[0] < 16:
                        break
                    
                    kmeans.partial_fit(out.T)

        cluster_labels = kmeans.labels_
        matches = torch.zeros((feature_dim, int(self.r * feature_dim)), device=self.device)
        for model_idx, match_idx in enumerate(cluster_labels):
            matches[model_idx, match_idx] = 1
        
        merge = matches / (matches.sum(dim=0, keepdim=True) + 1e-5)
        unmerge = matches.detach().clone()

        model.layers[layer_idx] = DecompAEWrapper(
            model.layers[layer_idx],
            ClusteringAE(model.layers[layer_idx].weight, model.layers[layer_idx].bias, merge, unmerge, self.device)
        )
        model.layers[layer_idx + 1] = torch.nn.Identity() 

        model.layers[layer_idx].autoencoder.weight.requires_grad = False
        model.layers[layer_idx].autoencoder.bias.requires_grad = False
        model.layers[layer_idx].autoencoder.merge.requires_grad = False

        optim = torch.optim.Adam(
            model.layers[layer_idx].autoencoder.parameters(),
            0.005,
        )
        loss_fn = torch.nn.MSELoss()             #55
        train_loop(model, layer_idx, optim, loss_fn, self.epochs, self.loader, self.device)

        return model.to(self.device)
    