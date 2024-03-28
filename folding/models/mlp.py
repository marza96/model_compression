import torch
import torch.nn as nn

from typing import Any, NamedTuple


class MLPSubnet(nn.Module):
    def __init__(self, model, layer_i):
        super().__init__()
        self.model = model
        self.layer_i = layer_i

    def forward(self, x):
        if x.size(1) == 3:
            x = x.mean(1, keepdim=True)

        x = x.reshape(x.size(0), -1)
        x = self.model.layers[:self.layer_i + 1](x)
        
        return x
    

class MLPSpec:
    def __init__(self, layers, bnorm=False):
        self._cfg               = layers
        self._layer_spec        = list()
        self._perm_spec         = list()
        self._layer_spec_unique = list()

        offset = 0
        for i in range(layers):
            modules = [
                f"layers.{offset}.weight",
                f"layers.{offset}.bias",
            ]
            perms = [
                (i, i - 1),
                (i, -1)
            ]
            unique_modules = [
                f"layers.{offset}"
            ]

            if bnorm is True:
                modules.extend(
                    [
                        f"layers.{offset + 1}.weight",
                        f"layers.{offset + 1}.bias",
                        f"layers.{offset + 1}.running_mean",
                        f"layers.{offset + 1}.running_var",
                    ]
                )
                perms.extend(
                    [
                        (i, -1),
                        (i, -1),
                        (i, -1),
                        (i, -1),
                    ]   
                )
                unique_modules.extend(
                    [
                        f"layers.{offset + 1}"
                    ]
                )

            self._layer_spec.append(modules)
            self._perm_spec.append(perms)
            self._layer_spec_unique.append(unique_modules)

            offset += 3 - (not bnorm)

        self._layer_spec.append(
            [
                f"layers.{offset}.weight",
                f"layers.{offset}.bias",
            ]
        )
        self._perm_spec.append(
            [
                (-1, layers - 1),
                (-1, -1)
            ]
        )
        self._layer_spec_unique.append(
            [f"layers.{offset + bnorm}"]
        )

        if bnorm is True:
            self._layer_spec[-1].extend(
                [
                    f"layers.{offset +  bnorm}.weight",
                    f"layers.{offset +  bnorm}.bias",
                    f"layers.{offset +  bnorm}.running_mean",
                    f"layers.{offset +  bnorm}.running_var",
                ]
            )
            self._perm_spec[-1].extend(
                [
                    (-1, -1),
                    (-1, -1),
                    (-1, -1),
                    (-1, -1),
                ]   
            )
            self._layer_spec_unique[-1].extend(
                [
                    f"layers.{offset + bnorm}"
                ]
            )

    @property
    def cfg(self):
        return self._cfg
    
    @property
    def layer_spec(self):
        return self._layer_spec
    
    @property
    def perm_spec(self):
        return self._perm_spec
    
    @property
    def layer_spec_unique(self):
        return self._layer_spec_unique


class MLP(nn.Module):
    def __init__(self, channels=128, layers=3, classes=10, bnorm=False):
        super().__init__()
        
        self.bnorm      = bnorm
        self.classes    = classes
        self.channels   = channels
        self.num_layers = layers
        self.subnet     = MLPSubnet
        self.spec  = MLPSpec(layers, bnorm=bnorm)
        
        mid_layers = [
            nn.Linear(28 * 28, channels, bias=True),
        ]
        if self.bnorm is True:
            mid_layers.append(nn.BatchNorm1d(channels))

        mid_layers.append(nn.ReLU())

        for i in range(layers):
            lst  = [
                nn.Linear(channels, channels, bias=True),
            ]
            if self.bnorm is True:
                lst.append(nn.BatchNorm1d(channels))
                
            lst.append(nn.ReLU())

            if i == self.num_layers - 1:
                lst = [
                    nn.Linear(channels, channels, bias=True),
                ]
                if self.bnorm is True:
                    lst.append(nn.BatchNorm1d(channels))

            mid_layers.extend(lst)
            
        self.layers = nn.Sequential(*mid_layers)

    def forward(self, x):
        if x.size(1) == 3:
            x = x.mean(1, keepdim=True)

        x = x.reshape(x.size(0), -1)

        x = self.layers(x)
 
        return x
    

if __name__ == "__main__":
    spec = MLPSpec(5, False)
    print(spec.layer_spec_unique)
    