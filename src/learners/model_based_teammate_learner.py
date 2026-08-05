"""PPO learner conditioned on a specialist predictive teammate model."""

import copy
import os

import torch as th
import torch.nn.functional as F
from torch.optim import Adam

from controllers import AgentOwnedMAC, OpenTrainMAC
from learners.ppo_learner import PPOLearner
from modules.predictive_teammate import PredictiveTeammateModel


class ModelBasedTeammateLearner(PPOLearner):
    def __init__(self, mac, scheme, logger, args):
        super().__init__(mac, scheme, logger, args)
        obs_dim = scheme["obs"]["vshape"]
        self.teammate_model = PredictiveTeammateModel(obs_dim, args)
        self.target_teammate_model = copy.deepcopy(self.teammate_model)
        self.model_params = list(self.teammate_model.parameters())
        self.model_optimiser = Adam(
            self.model_params,
            lr=getattr(args, "teammate_model_lr", 3e-4),
            eps=args.optim_eps,
        )
        for parameter in self.target_teammate_model.parameters():
            parameter.requires_grad_(False)
        self.target_teammate_model.eval()

        if isinstance(mac, AgentOwnedMAC):
            mac.agent.encoder = self.target_teammate_model
        elif isinstance(mac, OpenTrainMAC):
            mac.set_encoder(self.target_teammate_model)
        else:
            raise TypeError("Model-based teammate PPO requires an agent-owned controller")
        self.critic.encoder = self.target_teammate_model
        self.model_log_stats_t = 0

    def train(self, batch, t_env, episode_num):
        self._cache_contexts(batch)
        super().train(batch, t_env, episode_num)
        stats = self._model_update(batch)
        self._momentum_update_target()
        if t_env - self.model_log_stats_t >= self.args.learner_log_interval or self.model_log_stats_t == 0:
            for key, value in stats.items():
                self.logger.log_stat(key, value, t_env)
            self.model_log_stats_t = t_env

    @th.no_grad()
    def _cache_contexts(self, batch):
        valid = batch["filled"].squeeze(-1)
        batch.clam_contexts = self.target_teammate_model.forward_prefixes(
            batch["obs"], valid=valid
        ).detach()

    def _model_update(self, batch):
        obs = batch["obs"]
        actions = batch["actions_onehot"]
        rewards = batch["reward"]
        trainable = batch["trainable_agents"].squeeze(-1).bool()
        time = min(actions.size(1), obs.size(1) - 1, rewards.size(1))
        valid = batch["filled"][:, :time].squeeze(-1).bool()

        batch_size, _, n_agents, obs_dim = obs.shape
        trajectories = obs.permute(0, 2, 1, 3).reshape(
            batch_size * n_agents, obs.size(1), obs_dim
        )
        latent_all = self.teammate_model.encode_sequence(trajectories)
        latent_all = latent_all.view(batch_size, n_agents, obs.size(1), -1).permute(0, 2, 1, 3)
        latent = latent_all[:, :time]
        next_latent = latent_all[:, 1 : time + 1].detach()

        uncontrolled = (~trainable[:, :time]).float()
        teammate_denom = uncontrolled.sum(dim=2, keepdim=True).clamp_min(1.0)
        teammate_actions = (
            actions[:, :time] * uncontrolled.unsqueeze(-1)
        ).sum(dim=2) / teammate_denom
        teammate_targets = teammate_actions.unsqueeze(2).expand(-1, -1, n_agents, -1)

        controlled = trainable[:, :time]
        mask = controlled & valid.unsqueeze(-1)
        if not mask.any():
            return {"teammate_model_loss": 0.0}

        flat_latent = latent[mask]
        flat_next_latent = next_latent[mask]
        flat_ego_action = actions[:, :time][mask]
        flat_teammate_action = teammate_targets[mask]
        flat_reward = rewards[:, :time].unsqueeze(2).expand(-1, -1, n_agents, -1)[mask]
        flat_obs_delta = (obs[:, 1 : time + 1] - obs[:, :time])[mask]

        logits = self.teammate_model.teammate_action_logits(flat_latent)
        action_loss = -(flat_teammate_action * F.log_softmax(logits, dim=-1)).sum(-1).mean()
        predicted_next = self.teammate_model.transition(
            flat_latent, flat_ego_action, flat_teammate_action
        )
        dynamics_loss = F.smooth_l1_loss(predicted_next, flat_next_latent)
        reward_prediction = self.teammate_model.predict_reward(
            flat_latent, flat_ego_action, flat_teammate_action
        )
        reward_loss = F.smooth_l1_loss(reward_prediction, flat_reward)
        obs_prediction = self.teammate_model.predict_obs_delta(
            flat_latent, flat_ego_action, flat_teammate_action
        )
        obs_loss = F.smooth_l1_loss(obs_prediction, flat_obs_delta)

        loss = (
            getattr(self.args, "teammate_action_loss_coef", 1.0) * action_loss
            + getattr(self.args, "teammate_dynamics_loss_coef", 1.0) * dynamics_loss
            + getattr(self.args, "teammate_reward_loss_coef", 0.1) * reward_loss
            + getattr(self.args, "teammate_obs_loss_coef", 0.25) * obs_loss
        )
        self.model_optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(
            self.model_params, getattr(self.args, "teammate_model_grad_norm_clip", 5.0)
        )
        self.model_optimiser.step()

        target_class = flat_teammate_action.argmax(dim=-1)
        accuracy = (logits.argmax(dim=-1) == target_class).float().mean()
        return {
            "teammate_model_loss": loss.item(),
            "teammate_action_loss": action_loss.item(),
            "teammate_action_accuracy": accuracy.item(),
            "teammate_dynamics_loss": dynamics_loss.item(),
            "teammate_reward_loss": reward_loss.item(),
            "teammate_obs_loss": obs_loss.item(),
            "teammate_model_grad_norm": grad_norm.item(),
        }

    @th.no_grad()
    def _momentum_update_target(self):
        tau = getattr(self.args, "teammate_target_tau", 0.01)
        for online, target in zip(
            self.teammate_model.parameters(), self.target_teammate_model.parameters()
        ):
            target.data.mul_(1.0 - tau).add_(online.data, alpha=tau)
        for online, target in zip(
            self.teammate_model.buffers(), self.target_teammate_model.buffers()
        ):
            target.copy_(online)
        self.target_teammate_model.eval()

    def cuda(self):
        super().cuda()
        self.teammate_model.cuda()
        self.target_teammate_model.cuda()

    def save_models(self, path):
        super().save_models(path)
        th.save(self.teammate_model.state_dict(), os.path.join(path, "teammate_model.th"))
        th.save(self.target_teammate_model.state_dict(), os.path.join(path, "teammate_target_model.th"))
        th.save(self.model_optimiser.state_dict(), os.path.join(path, "teammate_model_opt.th"))

    def load_models(self, path):
        super().load_models(path)
        map_location = lambda storage, loc: storage
        self.teammate_model.load_state_dict(th.load(os.path.join(path, "teammate_model.th"), map_location=map_location))
        target_path = os.path.join(path, "teammate_target_model.th")
        if os.path.exists(target_path):
            self.target_teammate_model.load_state_dict(th.load(target_path, map_location=map_location))
        else:
            self.target_teammate_model.load_state_dict(self.teammate_model.state_dict())
        opt_path = os.path.join(path, "teammate_model_opt.th")
        if os.path.exists(opt_path):
            self.model_optimiser.load_state_dict(th.load(opt_path, map_location=map_location))
        self.target_teammate_model.eval()
