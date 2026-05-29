import torch
import torch.nn as nn


class UncontrolledLSTMClassifier(nn.Module):
    """LSTM classifier over controlled agents' observation histories."""

    def __init__(
        self,
        obs_dim: int,
        n_agents: int,
        episode_limit: int,
        num_uncontrolled_types: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()
        self.n_agents = n_agents
        self.obs_proj = nn.Linear(obs_dim + 1, hidden_dim)
        self.lstm = nn.LSTM(
            input_size=hidden_dim * n_agents,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
            batch_first=True,
        )
        output_dim = hidden_dim * (2 if bidirectional else 1)
        count_features = 2
        self.cls_head = nn.Sequential(
            nn.LayerNorm(output_dim + count_features),
            nn.Dropout(dropout),
            nn.Linear(output_dim + count_features, num_uncontrolled_types),
        )

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
        if nagents != self.n_agents:
            raise ValueError(f"Expected {self.n_agents} agents, got {nagents}")

        valid_agent_mask = time_mask.unsqueeze(2) & agent_mask
        obs_with_mask = torch.cat([obs, valid_agent_mask.to(dtype=obs.dtype)], dim=-1)
        x = self.obs_proj(obs_with_mask)
        x = (x * valid_agent_mask.to(dtype=x.dtype)).reshape(bsz, timesteps, nagents * x.shape[-1])

        encoded, _ = self.lstm(x)
        last_indices = torch.full(
            (encoded.shape[0],),
            encoded.shape[1] - 1,
            dtype=torch.long,
            device=encoded.device,
        )
        pooled = encoded[torch.arange(encoded.shape[0], device=encoded.device), last_indices]

        controlled_agents = valid_agent_mask.any(dim=1).squeeze(-1).sum(dim=1).to(dtype=pooled.dtype)
        controlled_frac = controlled_agents / float(self.n_agents)
        uncontrolled_frac = 1.0 - controlled_frac
        count_context = torch.stack([controlled_frac, uncontrolled_frac], dim=-1)
        return self.cls_head(torch.cat([pooled, count_context], dim=-1))
