"""Multi-Agent Transformer World Model components.

This is a vector-observation implementation of MATWM (Deihim et al., 2025)
adapted to EPyMARL's episode-major data layout.  The world model is shared,
but each focal agent is imagined independently and owns an actor and critic.
"""

import copy

import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


def _causal_mask(length, device):
    return th.triu(th.full((length, length), float("-inf"), device=device), diagonal=1)


class CategoricalEncoder(nn.Module):
    def __init__(self, obs_dim, hidden_dim, layers, n_latents, n_categories):
        super().__init__()
        blocks = []
        in_dim = obs_dim
        for _ in range(layers):
            blocks.extend((nn.Linear(in_dim, hidden_dim), nn.ELU()))
            in_dim = hidden_dim
        blocks.append(nn.Linear(in_dim, n_latents * n_categories))
        self.net = nn.Sequential(*blocks)
        self.n_latents = n_latents
        self.n_categories = n_categories

    def logits(self, obs):
        return self.net(obs).view(*obs.shape[:-1], self.n_latents, self.n_categories)

    def forward(self, obs, sample=True):
        logits = self.logits(obs)
        if sample:
            # Straight-through categorical sample used by STORM/MATWM.
            indices = Categorical(logits=logits).sample()
            hard = F.one_hot(indices, self.n_categories).to(logits.dtype)
            probs = logits.softmax(-1)
            latent = hard + probs - probs.detach()
        else:
            indices = logits.argmax(-1)
            latent = F.one_hot(indices, self.n_categories).to(logits.dtype)
        return latent, logits


class MATWMWorldModel(nn.Module):
    def __init__(self, obs_dim, n_agents, n_actions, args):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.n_latents = getattr(args, "matwm_n_latents", 32)
        self.n_categories = getattr(args, "matwm_n_categories", 32)
        self.latent_dim = self.n_latents * self.n_categories
        self.hidden_dim = getattr(args, "matwm_hidden_dim", 512)
        self.max_seq_length = getattr(args, "matwm_max_seq_length", 64)

        encoder_hidden = getattr(args, "matwm_encoder_hidden_dim", 512)
        encoder_layers = getattr(args, "matwm_encoder_layers", 3)
        self.encoder = CategoricalEncoder(
            self.obs_dim, encoder_hidden, encoder_layers,
            self.n_latents, self.n_categories,
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, encoder_hidden), nn.ELU(),
            nn.Linear(encoder_hidden, encoder_hidden), nn.ELU(),
            nn.Linear(encoder_hidden, self.obs_dim),
        )

        # Agent identity is represented by mutually orthogonal action blocks:
        # action a of focal agent i maps to i * |A| + a.
        self.action_mixer = nn.Linear(
            self.latent_dim + n_agents * n_actions, self.hidden_dim
        )
        self.position = nn.Parameter(th.zeros(1, self.max_seq_length, self.hidden_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=getattr(args, "matwm_attention_heads", 8),
            dim_feedforward=getattr(args, "matwm_ff_dim", self.hidden_dim * 4),
            dropout=getattr(args, "matwm_dropout", 0.0),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.sequence_model = nn.TransformerEncoder(
            layer, num_layers=getattr(args, "matwm_transformer_layers", 2),
            norm=nn.LayerNorm(self.hidden_dim),
        )
        self.dynamics = nn.Linear(self.hidden_dim, self.latent_dim)

        self.reward_bins = getattr(args, "matwm_reward_bins", 255)
        self.reward_low = getattr(args, "matwm_reward_low", -20.0)
        self.reward_high = getattr(args, "matwm_reward_high", 20.0)
        self.reward = nn.Linear(self.hidden_dim, self.reward_bins)
        self.continuation = nn.Linear(self.hidden_dim, 1)
        self.action_mask = nn.Linear(self.latent_dim, n_actions)

        teammate_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=getattr(args, "matwm_attention_heads", 8),
            dim_feedforward=getattr(args, "matwm_ff_dim", self.hidden_dim * 4),
            dropout=getattr(args, "matwm_dropout", 0.0),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.teammate_input = nn.Linear(self.latent_dim, self.hidden_dim)
        self.teammate_position = nn.Parameter(
            th.zeros(1, self.max_seq_length, self.hidden_dim)
        )
        self.teammate_model = nn.TransformerEncoder(
            teammate_layer,
            num_layers=getattr(args, "matwm_teammate_layers", 2),
            norm=nn.LayerNorm(self.hidden_dim),
        )
        self.teammate_head = nn.Linear(self.hidden_dim, n_agents * n_actions)

        nn.init.normal_(self.position, std=0.02)
        nn.init.normal_(self.teammate_position, std=0.02)

    @staticmethod
    def symlog(value):
        return th.sign(value) * th.log1p(value.abs())

    @staticmethod
    def symexp(value):
        return th.sign(value) * th.expm1(value.abs())

    def encode(self, obs, sample=True):
        return self.encoder(obs, sample=sample)

    def decode(self, latent):
        return self.decoder(latent.flatten(-2))

    def scaled_action(self, actions, focal_ids):
        actions = actions.long().squeeze(-1)
        shape = actions.shape + (self.n_agents * self.n_actions,)
        out = th.zeros(shape, device=actions.device, dtype=self.position.dtype)
        indices = focal_ids.long() * self.n_actions + actions
        return out.scatter_(-1, indices.unsqueeze(-1), 1.0)

    def dynamics_sequence(self, latent, actions, focal_ids):
        """Return h_t for tokens (z_t, a_t), with a causal attention mask."""
        length = latent.shape[1]
        if length > self.max_seq_length:
            latent = latent[:, -self.max_seq_length:]
            actions = actions[:, -self.max_seq_length:]
            length = self.max_seq_length
        ids = focal_ids[:, None].expand(-1, length)
        action = self.scaled_action(actions, ids)
        token = self.action_mixer(th.cat((latent.flatten(-2), action), dim=-1))
        token = token + self.position[:, :length]
        return self.sequence_model(token, mask=_causal_mask(length, token.device))

    def teammate_logits(self, latent, detach_encoder=True):
        length = latent.shape[1]
        if length > self.max_seq_length:
            latent = latent[:, -self.max_seq_length:]
            length = self.max_seq_length
        flat = latent.flatten(-2)
        if detach_encoder:
            flat = flat.detach()
        token = self.teammate_input(flat) + self.teammate_position[:, :length]
        hidden = self.teammate_model(token, mask=_causal_mask(length, token.device))
        return self.teammate_head(hidden).view(
            *hidden.shape[:-1], self.n_agents, self.n_actions
        )

    def prediction_heads(self, hidden):
        dynamics_logits = self.dynamics(hidden).view(
            *hidden.shape[:-1], self.n_latents, self.n_categories
        )
        return {
            "dynamics_logits": dynamics_logits,
            "reward_logits": self.reward(hidden),
            "continuation_logits": self.continuation(hidden),
        }

    def reward_value(self, reward_logits):
        bins = th.linspace(
            self.reward_low, self.reward_high, self.reward_bins,
            device=reward_logits.device, dtype=reward_logits.dtype,
        )
        return self.symexp((reward_logits.softmax(-1) * bins).sum(-1, keepdim=True))

    def two_hot_reward(self, reward):
        value = self.symlog(reward).clamp(self.reward_low, self.reward_high)
        position = (value - self.reward_low) / (self.reward_high - self.reward_low)
        position = position * (self.reward_bins - 1)
        lower = position.floor().long().clamp(0, self.reward_bins - 1)
        upper = position.ceil().long().clamp(0, self.reward_bins - 1)
        upper_weight = position - lower.to(position.dtype)
        target = th.zeros(
            *value.shape[:-1], self.reward_bins,
            device=value.device, dtype=value.dtype,
        )
        target.scatter_add_(-1, lower, 1.0 - upper_weight)
        target.scatter_add_(-1, upper, upper_weight)
        return target

    def predicted_availability(self, latent):
        logits = self.action_mask(latent.flatten(-2))
        available = logits.sigmoid() >= 0.5
        none_available = ~available.any(-1, keepdim=True)
        fallback = F.one_hot(logits.argmax(-1), self.n_actions).bool()
        return th.where(none_available, fallback, available)


class AgentMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class MATWMPolicy(nn.Module):
    """World model plus per-agent actors, critics, and EMA critics."""

    def __init__(self, obs_dim, n_agents, n_actions, args):
        super().__init__()
        self.args = args
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.world_model = MATWMWorldModel(obs_dim, n_agents, n_actions, args)
        state_dim = (
            self.world_model.latent_dim + self.world_model.hidden_dim
            + n_agents * n_actions
        )
        agent_hidden = getattr(args, "matwm_agent_hidden_dim", 512)
        self.actors = nn.ModuleList([
            AgentMLP(state_dim, agent_hidden, n_actions) for _ in range(n_agents)
        ])
        self.critics = nn.ModuleList([
            AgentMLP(state_dim, agent_hidden, 1) for _ in range(n_agents)
        ])
        self.ema_critics = copy.deepcopy(self.critics)
        for parameter in self.ema_critics.parameters():
            parameter.requires_grad_(False)

    @property
    def hidden_dim(self):
        return self.world_model.hidden_dim

    def actor_parameters(self):
        return self.actors.parameters()

    def critic_parameters(self):
        return self.critics.parameters()

    def agent_forward(self, modules, state, focal_ids):
        output_dim = modules[0](state[:1]).shape[-1]
        output = state.new_zeros(state.shape[0], output_dim)
        for agent_id, module in enumerate(modules):
            selected = focal_ids == agent_id
            if selected.any():
                output[selected] = module(state[selected])
        return output

    def actor_logits(self, state, focal_ids):
        return self.agent_forward(self.actors, state, focal_ids)

    def values(self, state, focal_ids, ema=False):
        modules = self.ema_critics if ema else self.critics
        return self.agent_forward(modules, state, focal_ids)

    def build_state(self, latent, hidden, teammate_logits, focal_ids):
        # Zero the focal agent's slot: only predicted non-focal actions are used.
        teammate_logits = teammate_logits.clone()
        rows = th.arange(focal_ids.shape[0], device=focal_ids.device)
        teammate_logits[rows, focal_ids] = 0.0
        return th.cat((
            latent.flatten(-2), hidden, teammate_logits.flatten(-2)
        ), dim=-1)

    def real_policy(self, batch, t, agent_indices, test_mode=False):
        """Policy logits/actions from the real local history at environment step t."""
        ts = t + 1
        obs = batch["obs"][:, :ts, agent_indices]
        bsz, length, count, _ = obs.shape
        obs = obs.permute(0, 2, 1, 3).reshape(bsz * count, length, -1)
        focal = th.tensor(agent_indices, device=obs.device).repeat(bsz)
        latent, _ = self.world_model.encode(obs, sample=not test_mode)

        if t == 0:
            hidden = obs.new_zeros(bsz * count, self.hidden_dim)
        else:
            previous_actions = batch["actions"][:, :t, agent_indices]
            previous_actions = previous_actions.permute(0, 2, 1, 3).reshape(
                bsz * count, t, 1
            )
            hidden = self.world_model.dynamics_sequence(
                latent[:, :-1], previous_actions, focal
            )[:, -1]

        teammate = self.world_model.teammate_logits(latent)[:, -1]
        state = self.build_state(latent[:, -1], hidden, teammate, focal)
        logits = self.actor_logits(state, focal).view(bsz, count, self.n_actions)
        return logits, hidden.view(bsz, count, self.hidden_dim)

    @th.no_grad()
    def update_ema(self, decay):
        for target, source in zip(self.ema_critics.parameters(), self.critics.parameters()):
            target.mul_(decay).add_(source, alpha=1.0 - decay)
