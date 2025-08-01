import torch
from torch import nn
import torch.nn.functional as F

class EstimationNetwork(nn.Module):
    def __init__(self, input_dim: int, output_dim: int,
                 hidden_dims: list[int], activation=nn.ReLU, dropout=0.5):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                activation(),
                nn.Dropout(dropout)
            ]
            prev = h
        layers += [nn.Linear(prev, output_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return F.softmax(self.net(x), dim=1)
