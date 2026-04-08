import math
import torch
import torch.nn as nn


class UncontrolledTransformerClassifier(nn.Module):
    """Transformer encoder that ingests controlled agents' observation histories."""

    def __init__(
        self,
        obs_dim: int,
        n_agents: int,
        episode_limit: int,
        num_uncontrolled_types: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.obs_proj = nn.Linear(obs_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.cls_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_uncontrolled_types),
        )
        self.register_buffer("time_pos_emb", self._positional_encoding(episode_limit + 1, d_model))
        self.register_buffer("agent_pos_emb", self._positional_encoding(n_agents, d_model))

    @staticmethod
    def _positional_encoding(length: int, d_model: int) -> torch.Tensor:
        """Sinusoidal embeddings."""
        position = torch.arange(length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(length, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(
        self,
        obs: torch.Tensor,
        time_mask: torch.Tensor,
        agent_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            obs: (B, T, A, obs_dim)
            time_mask: (B, T, 1) bool, True where timestep valid
            agent_mask: (B, T, A, 1) bool, True for controlled agents
        Returns:
            logits: (B, num_uncontrolled_types)
        """
        bsz, timesteps, nagents, _ = obs.shape
        x = self.obs_proj(obs)
        time_emb = self.time_pos_emb[:timesteps].view(1, timesteps, 1, -1)
        agent_emb = self.agent_pos_emb[:nagents].view(1, 1, nagents, -1)
        x = x + time_emb + agent_emb
        x = x.view(bsz, timesteps * nagents, -1)

        flat_mask = (time_mask & agent_mask.squeeze(-1)).view(bsz, timesteps * nagents)
        key_padding_mask = ~flat_mask
        encoded = self.encoder(x, src_key_padding_mask=key_padding_mask)

        denom = flat_mask.sum(dim=1, keepdim=True).clamp(min=1)
        pooled = (encoded * flat_mask.unsqueeze(-1)).sum(dim=1) / denom
        return self.cls_head(pooled)
