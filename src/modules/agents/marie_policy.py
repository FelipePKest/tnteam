"""MARIE policy facade combining world model, actor, and critic."""

import torch as th
import torch.nn as nn
import torch.nn.functional as F

import copy
from modules.matwm import MATWMPolicy
from modules.world_models.marie.world_model import MARIEWorldModel
from modules.agents.marie_actor import OriginalMARIEActor
from modules.critics.marie import OriginalMARIECritic

class MARIEPolicy(MATWMPolicy):
    """MARIE world model with the established per-agent actor/critic API."""

    def __init__(self, obs_dim, n_agents, n_actions, args):
        nn.Module.__init__(self)
        self.args = args
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.use_raw_obs_skip = False
        self.policy_obs_norm = None
        self.world_model = MARIEWorldModel(obs_dim, n_agents, n_actions, args)
        self.stack_obs = getattr(args, "marie_stack_obs", 5)
        self.actor_feature_dim = obs_dim * self.stack_obs
        self.critic_feature_dim = self.actor_feature_dim
        agent_hidden = getattr(args, "matwm_agent_hidden_dim", 256)
        # Upstream MARIE feeds the same reconstructed local observation stack
        # to actor and critic. The Perceiver state belongs to the environment.
        self.actors = nn.ModuleList([
            OriginalMARIEActor(self.actor_feature_dim, agent_hidden, n_actions)
        ])
        self.critics = nn.ModuleList([
            OriginalMARIECritic(
                self.critic_feature_dim, agent_hidden,
                getattr(args, "marie_critic_heads", 4),
            )
        ])
        self.ema_critics = copy.deepcopy(self.critics)
        for parameter in self.ema_critics.parameters():
            parameter.requires_grad_(False)
        # Match DreamerLearner's explicit StarCraft initialization: actor
        # orthogonal, critic Xavier (biases remain at their zero defaults).
        for parameter in self.actors.parameters():
            if parameter.ndim >= 2:
                nn.init.orthogonal_(parameter)
        for parameter in self.critics.parameters():
            if parameter.ndim >= 2:
                nn.init.xavier_uniform_(parameter)
        self.ema_critics.load_state_dict(self.critics.state_dict())

    def agent_forward(self, modules, state, focal_ids):
        return modules[0](state)

    def actor_logits(self, state, focal_ids):
        return self.actors[0](state[..., :self.actor_feature_dim])

    def values(self, state, focal_ids, ema=False):
        modules = self.ema_critics if ema else self.critics
        features = state[..., :self.actor_feature_dim]
        if features.dim() == 2 and features.shape[0] % self.n_agents == 0:
            ids = focal_ids.reshape(-1, self.n_agents)
            expected = th.arange(self.n_agents, device=ids.device)[None]
            if th.equal(ids, expected.expand_as(ids)):
                values = modules[0](
                    features.view(-1, self.n_agents, features.shape[-1])
                )
                return values.reshape(features.shape[0], 1)
        return modules[0].independent(features)

    def build_state(
        self, latent, hidden, teammate_logits, focal_ids, observation=None
    ):
        if observation is None:
            observation = self.world_model.decode(latent)
        else:
            # Execution and imagined learning must consume the same tokenizer
            # reconstruction distribution used by upstream MARIE.
            observation = self.reconstruct_observations(observation)
        return self.build_state_from_reconstructed(observation)

    @th.no_grad()
    def reconstruct_observations(self, observation):
        """Reconstruct each frame independently with the frozen tokenizer."""
        obs_latent, _ = self.world_model.encode(observation, sample=False)
        return self.world_model.decode(obs_latent)

    def build_state_from_reconstructed(self, observation):
        """Format already-reconstructed frames without another tokenizer pass."""
        # MAWorldModelEnv clamps decoded SMAC observations before both acting
        # and critic evaluation.
        observation = observation.clamp(-1.0, 1.0)
        if observation.dim() == 2:
            observation = observation[:, None]
        observation = observation[:, -self.stack_obs:]
        if observation.shape[1] < self.stack_obs:
            padding = observation.new_zeros(
                observation.shape[0],
                self.stack_obs - observation.shape[1],
                observation.shape[-1],
            )
            observation = th.cat((padding, observation), dim=1)
        actor_features = observation.reshape(observation.shape[0], -1)
        return actor_features
