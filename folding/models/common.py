import torch
import torch.nn as nn


class SigmaWrapper(torch.nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer
        self.hess  = None
        self.cnt   = None

    def get_stats(self):
        assert self.hess is not None
        
        return self.hess / self.cnt

    def forward(self, x):
        hess = torch.multiply(x.unsqueeze(-1), x.unsqueeze(1)).sum(0)
        hess += torch.eye(hess.shape[0]).to(x.device) * 0.0001

        if self.hess is None:
            self.cnt  = 0
            self.hess = torch.zeros_like(hess)

        self.hess += hess
        self.cnt += x.shape[0]

        x = self.layer(x)

        return x
    

class CovWrapper(torch.nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer
        self.cov   = None

    def get_stats(self):
        assert self.cov is not None
        
        return self.cov

    def forward(self, x):
        self.cov = torch.cov(x)

        x = self.layer(x)

        return x
    

class LayerWrapper(nn.Module):
    def __init__(self, layer, rescale=False, w=False):
        super().__init__()
        self.layer   = layer
        self.rescale = rescale

        if w is True:
            self.bn = nn.BatchNorm1d(len(layer.layer.weight))
            self.bn.to(self.layer.layer.weight.device)
        else:
            self.bn = nn.BatchNorm1d(len(layer.weight))
            self.bn.to(self.layer.weight.device)

    def get_stats(self):
        mean = self.bn.running_mean
        std  = self.bn.running_var

        return mean, std
    
    def set_stats(self, mean, var):
        self.bn.bias.data = mean
        self.bn.weight.data = (var + 1e-4).sqrt()

    def forward(self, x):
        x = self.layer(x)
        x_rescaled = self.bn(x)

        if self.rescale is True:
            return x_rescaled
        
        return x
    

class LayerWrapper2D(nn.Module):
    def __init__(self, layer, rescale=False, w=False):
        super().__init__()
        self.layer   = layer
        self.rescale = rescale

        if w is True:
            self.bn = nn.BatchNorm2d(self.layer.layer.weight.shape[0])
            self.bn.to(self.layer.layer.weight.device)
        else:
            self.bn = nn.BatchNorm2d(layer.weight.shape[0])
            self.bn.to(self.layer.weight.device)

    def get_stats(self):
        mean = self.bn.running_mean
        std  = self.bn.running_var

        return mean, std
    
    def set_stats(self, mean, var):
        self.bn.bias.data = mean
        self.bn.weight.data = (var + 1e-7).sqrt()

    def forward(self, x):
        x = self.layer(x)
        x_rescaled = self.bn(x)

        if self.rescale is True:
            return x_rescaled
        
        return x