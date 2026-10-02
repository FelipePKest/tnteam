"""Perceiver aggregation over agents in each team."""

import torch as th
import torch.nn as nn
import torch.nn.functional as F


class _GEGLU(nn.Module):
    def forward(self, value):
        value, gate = value.chunk(2, dim=-1)
        return value * F.gelu(gate)

class _PerceiverAttention(nn.Module):
    def __init__(self, query_dim, context_dim, heads, dim_head, dropout):
        super().__init__()
        inner = heads * dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_q = nn.Linear(query_dim, inner, bias=True)
        self.to_kv = nn.Linear(context_dim, inner * 2, bias=True)
        self.to_out = nn.Linear(inner, query_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, context):
        batch, query_length = query.shape[:2]
        context_length = context.shape[1]
        q = self.to_q(query).view(
            batch, query_length, self.heads, -1
        ).transpose(1, 2)
        k, v = self.to_kv(context).chunk(2, dim=-1)
        k = k.view(batch, context_length, self.heads, -1).transpose(1, 2)
        v = v.view(batch, context_length, self.heads, -1).transpose(1, 2)
        attention = (q @ k.transpose(-2, -1) * self.scale).softmax(-1)
        output = self.dropout(attention) @ v
        output = output.transpose(1, 2).reshape(batch, query_length, -1)
        return self.to_out(output)

class _PerceiverFeedForward(nn.Module):
    def __init__(self, dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * 8), _GEGLU(),
            nn.Linear(dim * 4, dim), nn.Dropout(dropout),
        )

    def forward(self, value):
        return self.net(value)

class PerceiverAggregator(nn.Module):
    """Produce one centralized latent for every decentralized agent stream.

    Official MARIE uses ``NUM_AGENTS`` Perceiver latents.  Keeping one query
    per agent is important: a single pooled token makes every agent receive
    exactly the same centralized feature and is not the architecture described
    in the paper or implemented upstream.
    """

    def __init__(
        self, hidden_dim, n_agents, cross_heads, latent_heads, layers, dropout
    ):
        super().__init__()
        self.query = nn.Parameter(th.randn(n_agents, hidden_dim))
        self.cross_query_norm = nn.LayerNorm(hidden_dim)
        self.cross_context_norm = nn.LayerNorm(hidden_dim)
        # Reference configuration uses dim_head=64 independently of model dim.
        self.cross_attention = _PerceiverAttention(
            hidden_dim, hidden_dim, cross_heads, 64, dropout
        )
        self.cross_ff_norm = nn.LayerNorm(hidden_dim)
        self.cross_ff = _PerceiverFeedForward(hidden_dim, dropout)
        self.latent_attention = nn.ModuleList([
            _PerceiverAttention(
                hidden_dim, hidden_dim, latent_heads, 64, dropout
            ) for _ in range(layers)
        ])
        self.latent_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(layers)
        ])
        self.latent_ff = nn.ModuleList([
            _PerceiverFeedForward(hidden_dim, dropout) for _ in range(layers)
        ])
        self.latent_ff_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(layers)
        ])

    def forward(self, agent_features, n_agents=None):
        # agent_features: [batch*time, agents, hidden]
        queries = self.query if n_agents is None else self.query[:n_agents]
        token = queries.unsqueeze(0).expand(agent_features.shape[0], -1, -1)
        update = self.cross_attention(
            self.cross_query_norm(token),
            self.cross_context_norm(agent_features),
        )
        token = token + update
        token = token + self.cross_ff(self.cross_ff_norm(token))
        for attention, norm, feed_forward, ff_norm in zip(
            self.latent_attention, self.latent_norms, self.latent_ff,
            self.latent_ff_norms,
        ):
            normalized = norm(token)
            update = attention(normalized, normalized)
            token = token + update
            token = token + feed_forward(ff_norm(token))
        return token
