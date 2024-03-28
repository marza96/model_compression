import torch
import torch.nn as nn
import torch.nn.functional as F



class VGGSubnet(nn.Module):
    def __init__(self, model, layer_i):
        super().__init__()
        self.model = model
        self.layer_i = layer_i

    def forward(self, x):
        layer_type = str(type(self.model.layers[self.layer_i]))
        
        for i in range(self.layer_i + 1):
            layer_type = str(type(self.model.layers[i]))

            if "Linear" in  layer_type or "BatchNorm1d" in layer_type:
                x = x.reshape(x.size(0), -1)
            
            x = self.model.layers[i](x)

        return x
    

class VGGSpec:
    def __init__(self, cfg, bnorm=False):
        self._cfg               = cfg
        self._layer_spec        = list()
        self._perm_spec         = list()
        self._layer_spec_unique = list()

        offset = 0
        i      = 0
        for c in self._cfg:
            if c == "M":
                offset += 1
                continue

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

            i += 1
        
        self._layer_spec.append(
            [
                f"layers.{offset + 1}.weight",
                f"layers.{offset + 1}.bias",
            ]
        )
        self._perm_spec.append(
            [
                (-1, i - 1),
                (-1, -1)
            ]
        )
        self._layer_spec_unique.append(
            [f"layers.{offset + bnorm}"] 
        )

        if bnorm is True:
            self._layer_spec[-1].extend(
                [
                    f"layers.{offset + 1 + bnorm}.weight",
                    f"layers.{offset + 1 + bnorm}.bias",
                    f"layers.{offset + 1 + bnorm}.running_mean",
                    f"layers.{offset + 1 + bnorm}.running_var",
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
                    f"layers.{offset + 1 + bnorm}"
                ]
            )

    @property
    def cfg(self):
        return self._cfg
    
    @property
    def layer_spec(self):
        return self._layer_spec
    
    @property
    def layer_spec_unique(self):
        return self._layer_spec_unique
    
    @property
    def perm_spec(self):
        return self._perm_spec


class VGG(nn.Module):
    def __init__(self, cfg, w=1, classes=10, in_channels=3, bnorm=False):
        super().__init__()

        self.in_channels = in_channels
        self.w           = w
        self.bnorm       = bnorm
        self.classes     = classes
        self.subnet      = VGGSubnet
        self.spec   = VGGSpec(cfg, bnorm=bnorm)
        self.layers      = self._make_layers(cfg)

    def forward(self, x):
        out = self.layers[:-2](x)
        out = out.view(out.size(0), -1)
        out = self.layers[-2:](out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = self.in_channels

        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers.append(nn.Conv2d(in_channels if in_channels == 3 else self.w*in_channels,
                                     self.w*x, kernel_size=3, padding=1))
                
                if self.bnorm is True:
                    layers.append(nn.BatchNorm2d(self.w*x))

                layers.append(nn.ReLU(inplace=True))
                in_channels = x

        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        layers += [nn.Linear(self.w * cfg[-2], self.classes)]

        if self.bnorm is True:
            layers.append(nn.BatchNorm1d(self.classes))

        return nn.Sequential(*layers)