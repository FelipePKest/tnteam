"""Predictive teammate model with short-horizon latent rollouts.

The model only consumes an agent's local observation history at execution.
Privileged teammate actions, rewards, and next observations are used by the
learner as training targets, never as policy inputs.
"""

import torch as th
import torch.nn as nn
import torch.nn.functional as F


class PredictiveTeammateModel(nn.Module):
    def __init__(self, obs_dim, args):
        super().__init__()
        self.obs_dim = obs_dim
        self.n_actions = args.n_actions
        self.embed_dim = args.embed_dim
        self.horizon = getattr(args, "teammate_planning_horizon", 3)
        self.discount = getattr(args, "teammate_planning_discount", 0.95)
        hidden_dim = getattr(args, "teammate_model_hidden_dim", 128)

        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ReLU(),
        )
        self.history = nn.GRU(hidden_dim, self.embed_dim, batch_first=True)
        self.action_head = nn.Sequential(
            nn.Linear(self.embed_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, self.n_actions),
        )
        joint_action_dim = 2 * self.n_actions
        self.latent_dynamics = nn.GRUCell(joint_action_dim, self.embed_dim)
        predictor_input = self.embed_dim + joint_action_dim
        self.reward_head = nn.Sequential(
            nn.Linear(predictor_input, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1),
        )
        self.obs_delta_head = nn.Sequential(
            nn.Linear(predictor_input, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, obs_dim),
        )

    def encode_sequence(self, trajectories):
        """Return a latent for every prefix of ``[batch, time, obs]``."""
        features = self.obs_encoder(trajectories)
        latents, _ = self.history(features)
        return latents

    def teammate_action_logits(self, latents):
        return self.action_head(latents)

    def transition(self, latent, ego_action, teammate_action):
        joint_action = th.cat([ego_action, teammate_action], dim=-1)
        return self.latent_dynamics(joint_action, latent)

    def predict_reward(self, latent, ego_action, teammate_action):
        inputs = th.cat([latent, ego_action, teammate_action], dim=-1)
        return self.reward_head(inputs)

    def predict_obs_delta(self, latent, ego_action, teammate_action):
        inputs = th.cat([latent, ego_action, teammate_action], dim=-1)
        return self.obs_delta_head(inputs)

    def imagined_action_returns(self, latent):
        """Roll out every candidate ego action in latent space.

        Future ego actions repeat the initial candidate. This deliberately
        small, stable planner avoids compounding a learned policy inside the
        world model while still exposing action-conditioned consequences.
        """
        leading = latent.shape[:-1]
        candidates = th.eye(
            self.n_actions, dtype=latent.dtype, device=latent.device
        )
        candidates = candidates.view(
            *((1,) * len(leading)), self.n_actions, self.n_actions
        ).expand(*leading, -1, -1)
        imagined = latent.unsqueeze(-2).expand(*leading, self.n_actions, -1)
        returns = latent.new_zeros(*leading, self.n_actions)
        discount = 1.0
        for _ in range(self.horizon):
            teammate = F.softmax(self.action_head(imagined), dim=-1)
            reward = self.predict_reward(imagined, candidates, teammate).squeeze(-1)
            returns = returns + discount * reward
            imagined = self.transition(
                imagined.reshape(-1, self.embed_dim),
                candidates.reshape(-1, self.n_actions),
                teammate.reshape(-1, self.n_actions),
            ).view(*leading, self.n_actions, self.embed_dim)
            discount *= self.discount
        return returns

    def policy_context(self, latent):
        """Combine predictive belief and explicit action-lookahead features."""
        returns = th.tanh(self.imagined_action_returns(latent))
        if self.embed_dim < self.n_actions:
            raise ValueError("embed_dim must be at least n_actions")
        context = latent.clone()
        context[..., -self.n_actions :] = returns
        return context

    def forward_prefixes(self, observations, valid=None):
        batch, time, agents, obs_dim = observations.shape
        trajectories = observations.permute(0, 2, 1, 3).reshape(
            batch * agents, time, obs_dim
        )
        latents = self.encode_sequence(trajectories)
        contexts = self.policy_context(latents)
        return contexts.view(batch, agents, time, self.embed_dim).permute(0, 2, 1, 3)

    def forward(self, trajectories, padding_mask=None):
        latents = self.encode_sequence(trajectories)
        if padding_mask is None:
            final = latents[:, -1]
        else:
            lengths = (~padding_mask).long().sum(dim=1).clamp(min=1)
            final = latents[
                th.arange(latents.size(0), device=latents.device), lengths - 1
            ]
        return self.policy_context(final)
