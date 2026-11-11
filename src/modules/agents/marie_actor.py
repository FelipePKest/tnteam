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


class LatentMARIEActor(nn.Module):
    """Decentralized actions from ordered latent tokens, with separate padding."""
    def __init__(self, history_tokens, categories, hidden_dim, actions, embedding_dim=32):
        super().__init__()
        self.token_embedding = nn.Embedding(categories+1, embedding_dim, padding_idx=0)
        self.network = OriginalMARIEActor(history_tokens*embedding_dim,hidden_dim,actions)

    def forward(self, codes):
        return self.network(self.token_embedding(codes.long()+1).flatten(-2))
