import torch
from torch import nn


class EuclideanModel(nn.Module):
    def __init__(self, n_dimensions, weights=None, phi=None, p=2):
        super(EuclideanModel, self).__init__()
        self.dimension_weights = nn.Parameter(torch.empty(n_dimensions).uniform_(0, 1))
        # self.phi = nn.Parameter(torch.empty(1).uniform_(1, 100))
        # self.delta = nn.Parameter(torch.empty(1).uniform_(-1, 1))
        self.phi = torch.tensor(1.0)
        self.delta = torch.tensor(-1.0)
        self.p = p

    def forward(self, x, y):
        dist = (torch.norm((x - y) * self.dimension_weights, self.p, dim=2) / self.phi)
        return torch.sigmoid(self.delta) * torch.exp(-(dist ** 2))




