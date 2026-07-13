import math

import torch as th
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positions that also work for odd model dimensions."""

    def __init__(self, model_dim, max_len):
        super().__init__()
        position = th.arange(max_len, dtype=th.float32).unsqueeze(1)
        div_term = th.exp(
            th.arange(0, model_dim, 2, dtype=th.float32)
            * (-math.log(10000.0) / model_dim)
        )
        encoding = th.zeros(max_len, model_dim, dtype=th.float32)
        encoding[:, 0::2] = th.sin(position * div_term)
        encoding[:, 1::2] = th.cos(position * div_term[: model_dim // 2])
        self.register_buffer("encoding", encoding.unsqueeze(0))

    def forward(self, sequence):
        if sequence.size(1) > self.encoding.size(1):
            raise ValueError(
                "CLAM received a trajectory longer than its positional encoding "
                f"({sequence.size(1)} > {self.encoding.size(1)})"
            )
        return sequence + self.encoding[:, : sequence.size(1)]


class AttentionPooling(nn.Module):
    """Pool a feature sequence with the learned policy token from CLAM."""

    def __init__(self, model_dim, embed_dim, n_heads, dropout):
        super().__init__()
        self.policy_token = nn.Parameter(th.empty(1, 1, model_dim))
        nn.init.xavier_uniform_(self.policy_token)
        self.attention = nn.MultiheadAttention(
            model_dim,
            n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(model_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.ReLU(),
            nn.Linear(model_dim, embed_dim),
        )

    def forward(self, features, padding_mask=None):
        query = self.policy_token.expand(features.size(0), -1, -1)
        pooled, _ = self.attention(
            query,
            features,
            features,
            key_padding_mask=padding_mask,
            need_weights=False,
        )
        pooled = self.norm(pooled + query)
        return self.feed_forward(pooled[:, 0])

    def forward_prefixes(self, features, padding_mask=None):
        """Pool all prefixes without allowing a query to see future keys."""
        max_t = features.size(1)
        query = self.policy_token.expand(features.size(0), max_t, -1)
        future_mask = th.triu(
            th.ones(max_t, max_t, dtype=th.bool, device=features.device),
            diagonal=1,
        )
        pooled, _ = self.attention(
            query,
            features,
            features,
            key_padding_mask=padding_mask,
            attn_mask=future_mask,
            need_weights=False,
        )
        pooled = self.norm(pooled + query)
        return self.feed_forward(pooled)


class CLAMEncoder(nn.Module):
    """Transformer and attention-pooling policy encoder from CLAM.

    Inputs are local observation trajectories with shape ``[batch, time,
    obs_dim]``. ``forward_prefixes`` applies the exact same encoder to every
    trajectory prefix, matching real-time execution without exposing future
    observations.
    """

    def __init__(self, obs_dim, args):
        super().__init__()
        model_dim = getattr(args, "clam_model_dim", 128)
        n_heads = getattr(args, "clam_n_heads", 4)
        n_layers = getattr(args, "clam_n_layers", 2)
        ff_dim = getattr(args, "clam_ff_dim", model_dim * 2)
        dropout = getattr(args, "clam_dropout", 0.0)
        max_len = getattr(args, "episode_limit", 200) + 1

        if model_dim % n_heads != 0:
            raise ValueError("clam_model_dim must be divisible by clam_n_heads")

        self.embed_dim = args.embed_dim
        self.input_projection = nn.Linear(obs_dim, model_dim)
        self.position = SinusoidalPositionalEncoding(model_dim, max_len)
        layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="relu",
            batch_first=True,
            norm_first=False,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.pooling = AttentionPooling(
            model_dim=model_dim,
            embed_dim=args.embed_dim,
            n_heads=n_heads,
            dropout=dropout,
        )

    def forward(self, trajectory, padding_mask=None):
        if trajectory.dim() != 3:
            raise ValueError("CLAM trajectories must have shape [batch, time, obs_dim]")
        if trajectory.size(1) == 0:
            return trajectory.new_zeros(trajectory.size(0), self.embed_dim)

        if padding_mask is not None:
            padding_mask = padding_mask.bool().clone()
            # Multi-head attention cannot consume a sequence where every token
            # is padding. Keep the first (zero-valued) token available.
            all_padding = padding_mask.all(dim=1)
            padding_mask[all_padding, 0] = False

        features = self.position(self.input_projection(trajectory))
        max_t = trajectory.size(1)
        causal_mask = th.triu(
            th.ones(max_t, max_t, dtype=th.bool, device=trajectory.device),
            diagonal=1,
        )
        features = self.transformer(
            features,
            mask=causal_mask,
            src_key_padding_mask=padding_mask,
        )
        context = self.pooling(features, padding_mask=padding_mask)
        return F.normalize(context, dim=-1)

    def forward_prefixes(self, observations, valid=None):
        """Encode every prefix of per-agent observation histories.

        Args:
            observations: ``[batch, time, agents, obs_dim]``.
            valid: optional ``[batch, time]`` mask for padded episode steps.
        Returns:
            Policy contexts with shape ``[batch, time, agents, embed_dim]``.
        """
        batch_size, max_t, n_agents, obs_dim = observations.shape
        trajectories = observations.permute(0, 2, 1, 3).reshape(
            batch_size * n_agents, max_t, obs_dim
        )
        padding_mask = None
        if valid is not None:
            padding_mask = (~valid.bool()).unsqueeze(1).expand(-1, n_agents, -1)
            padding_mask = padding_mask.reshape(batch_size * n_agents, max_t)
            all_padding = padding_mask.all(dim=1)
            padding_mask[all_padding, 0] = False

        features = self.position(self.input_projection(trajectories))
        causal_mask = th.triu(
            th.ones(max_t, max_t, dtype=th.bool, device=observations.device),
            diagonal=1,
        )
        features = self.transformer(
            features,
            mask=causal_mask,
            src_key_padding_mask=padding_mask,
        )
        contexts = self.pooling.forward_prefixes(
            features, padding_mask=padding_mask
        )
        contexts = F.normalize(contexts, dim=-1)
        return contexts.view(batch_size, n_agents, max_t, self.embed_dim).permute(
            0, 2, 1, 3
        )


class CLAMProjectionHead(nn.Module):
    """Projection MLP used only by the contrastive objective."""

    def __init__(self, embed_dim, args):
        super().__init__()
        hidden_dim = getattr(args, "clam_projection_hidden_dim", 128)
        output_dim = getattr(args, "clam_projection_dim", 64)
        self.network = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, context):
        return F.normalize(self.network(context), dim=-1)


def symmetric_info_nce(first, second, temperature):
    """Symmetric InfoNCE over two aligned batches of normalized vectors."""
    if first.shape != second.shape:
        raise ValueError("InfoNCE views must have identical shapes")
    if first.size(0) < 2:
        raise ValueError("InfoNCE requires at least two trajectories")
    logits = first @ second.transpose(0, 1) / temperature
    labels = th.arange(first.size(0), device=first.device)
    return 0.5 * (
        F.cross_entropy(logits, labels)
        + F.cross_entropy(logits.transpose(0, 1), labels)
    )
