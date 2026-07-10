import math

import torch.nn as nn

from rfsensing.models import register


@register("mlp")
class MLP(nn.Module):
    """Flatten + fully-connected classifier."""

    def __init__(self, in_shape, num_classes, hidden_dims=(512, 256), dropout=0.5):
        super().__init__()
        dims = [math.prod(in_shape), *hidden_dims]
        layers: list[nn.Module] = [nn.Flatten()]
        for d_in, d_out in zip(dims, dims[1:]):
            layers += [nn.Linear(d_in, d_out), nn.ReLU(), nn.Dropout(dropout)]
        layers.append(nn.Linear(dims[-1], num_classes))
        self.net = nn.Sequential(*layers)

    def embed(self, x):
        return self.net[:-1](x)

    @property
    def head(self):
        return self.net[-1]

    def forward(self, x):
        return self.head(self.embed(x))