"""Team-attention value network used by MARIE."""

import torch as th
import torch.nn as nn
import torch.nn.functional as F


class OriginalMARIECritic(nn.Module):
    """Team-attention value model equivalent to upstream AugmentedCritic."""

    def __init__(self, input_dim, hidden_dim, heads):
        super().__init__()
        del heads  # Kept in the constructor for checkpoint/config compatibility.
        self.embed = nn.Linear(input_dim, hidden_dim)
        position = th.arange(30, dtype=th.float32)[:, None]
        dimension = th.arange(hidden_dim, dtype=th.float32)[None]
        angle = position / th.pow(
            10000.0, 2 * th.div(dimension, 2, rounding_mode="floor")
            / hidden_dim,
        )
        positional = th.empty(30, hidden_dim)
        positional[:, 0::2] = th.sin(angle[:, 0::2])
        positional[:, 1::2] = th.cos(angle[:, 1::2])
        self.register_buffer("positional", positional[None])
        # This is the exact AttentionEncoder configuration used by upstream
        # AugmentedCritic: one post-norm layer, eight heads, FF width=hidden.
        self.attention = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=hidden_dim,
                dropout=0.0,
                batch_first=True,
                norm_first=False,
            ),
            num_layers=1,
            enable_nested_tensor=False,
        )
        self.value = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, team_features):
        embedded = F.relu(self.embed(team_features))
        embedded = embedded + self.positional[:, :embedded.shape[1]]
        attended = self.attention(embedded)
        return self.value(F.relu(attended))

    def independent(self, features):
        return self.value(F.relu(self.embed(features)))
