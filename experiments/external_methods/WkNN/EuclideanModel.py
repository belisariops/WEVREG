import torch
from torch import nn
import numpy as np


class EuclideanModel(nn.Module):
    def __init__(self, n_dimensions, weights=None, phi=None):
        super(EuclideanModel, self).__init__()
        self.dimension_weights = nn.Parameter(torch.empty(n_dimensions).uniform_(1, 1))
        self.phi = torch.tensor(10.0)

    def forward(self, x, y):
        dist = (torch.norm((x - y) * self.dimension_weights, 2, dim=2))
        return torch.exp(-(dist**2) / self.phi )



