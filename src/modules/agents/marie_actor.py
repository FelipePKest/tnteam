"""Decentralized actor used by MARIE."""

import torch as th
import torch.nn as nn
import torch.nn.functional as F


class OriginalMARIEActor(nn.Module):
    """Two-hidden-layer ReLU actor used by upstream discrete MARIE."""

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, features):
        return self.net(features)
