"""Learner for the Multi-Agent Transformer World Model (MATWM)."""

import os

import torch as th
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.optim import Adam


class MATWMLearner:
    def __init__(self, mac, scheme, logger, args):
        self.mac = mac
        self.args = args
        self.logger = logger
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        default_agent_batch = 512 if self.n_agents <= 3 else (768 if self.n_agents <= 6 else 1024)
        default_horizon = 16 if self.n_agents <= 3 else (12 if self.n_agents <= 6 else 8)
        self.agent_batch_size = getattr(args, "matwm_agent_batch_size", None) or default_agent_batch
        self.imagination_horizon = getattr(args, "matwm_imagination_horizon", None) or default_horizon
        self.policy = self._policy_from_mac(mac)
        self.world_model = self.policy.world_model

        self.world_optimiser = Adam(
            self.world_model.parameters(),
            lr=getattr(args, "matwm_world_lr", 3e-5),
            eps=getattr(args, "optim_eps", 1e-5),
        )
        agent_parameters = list(self.policy.actor_parameters()) + list(
            self.policy.critic_parameters()
        )
        self.agent_optimiser = Adam(
            agent_parameters,
            lr=getattr(args, "matwm_agent_lr", 3e-4),
            eps=getattr(args, "optim_eps", 1e-5),
        )
        self.last_log_t = -getattr(args, "learner_log_interval", 10000) - 1

    @staticmethod
    def _policy_from_mac(mac):
        if hasattr(mac, "policy"):
            return mac.policy
        if hasattr(mac, "trained_agent") and hasattr(mac.trained_agent, "policy"):
            return mac.trained_agent.policy
        raise TypeError("MATWM requires MATWMMAC or MATWMTrainAgentLoader")

    @staticmethod
    def _masked_mean(value, mask):
        while mask.dim() < value.dim():
            mask = mask.unsqueeze(-1)
        return (value * mask).sum() / mask.expand_as(value).sum().clamp_min(1.0)

    def train(self, batch, t_env, episode_num):
        world_stats = self._train_world_model(batch)
        agent_stats = self._train_agents(batch)

        if t_env - self.last_log_t >= self.args.learner_log_interval:
            for key, value in {**world_stats, **agent_stats}.items():
                self.logger.log_stat(key, value, t_env)
            self.last_log_t = t_env

    def _train_world_model(self, batch):
        obs = batch["obs"]
        actions = batch["actions"]
        available = batch["avail_actions"].float()
        rewards = batch["reward"]
        terminated = batch["terminated"].float()
        valid = batch["filled"][:, :-1].float()
        bsz, total_t, n_agents, obs_dim = obs.shape
        length = total_t - 1
        if length < 1:
            return {"matwm_world_loss": 0.0}

        # The paper trains on non-overlapping sequences of at most 64 steps.
        # Episode replay can be longer, so select a contiguous window here.
        max_length = self.world_model.max_seq_length
        if length > max_length:
            start = int(th.randint(
                0, length - max_length + 1, (1,), device=obs.device
            ).item())
            stop = start + max_length
            obs = obs[:, start:stop + 1]
            actions = actions[:, start:stop + 1]
            available = available[:, start:stop + 1]
            rewards = rewards[:, start:stop]
            terminated = terminated[:, start:stop]
            valid = valid[:, start:stop]
            total_t = max_length + 1
            length = max_length

        # Each local trajectory becomes a focal-agent sequence for the shared WM.
        focal = th.arange(n_agents, device=obs.device).repeat(bsz)
        focal_obs = obs.permute(0, 2, 1, 3).reshape(bsz * n_agents, total_t, obs_dim)
        focal_actions = actions.permute(0, 2, 1, 3).reshape(
            bsz * n_agents, total_t, 1
        )
        focal_available = available.permute(0, 2, 1, 3).reshape(
            bsz * n_agents, total_t, self.n_actions
        )
        focal_valid = valid[:, None].expand(-1, n_agents, -1, -1).reshape(
            bsz * n_agents, length, 1
        )

        latent, posterior_logits = self.world_model.encode(focal_obs, sample=True)
        reconstruction = self.world_model.decode(latent)
        hidden = self.world_model.dynamics_sequence(
            latent[:, :-1], focal_actions[:, :-1], focal
        )
        heads = self.world_model.prediction_heads(hidden)

        reconstruction_loss = (reconstruction[:, :-1] - focal_obs[:, :-1]).pow(2).mean(-1, keepdim=True)

        reward_target = rewards[:, :length, None].expand(-1, -1, n_agents, -1)
        reward_target = reward_target.permute(0, 2, 1, 3).reshape(
            bsz * n_agents, length, 1
        )
        two_hot = self.world_model.two_hot_reward(reward_target)
        reward_loss = -(two_hot * heads["reward_logits"].log_softmax(-1)).sum(-1, keepdim=True)

        continue_target = (1.0 - terminated[:, :length])[:, None].expand(
            -1, n_agents, -1, -1
        ).reshape(bsz * n_agents, length, 1)
        continuation_loss = F.binary_cross_entropy_with_logits(
            heads["continuation_logits"], continue_target, reduction="none"
        )
        mask_loss = F.binary_cross_entropy_with_logits(
            self.world_model.action_mask(latent[:, :-1].flatten(-2)),
            focal_available[:, :-1], reduction="none"
        ).mean(-1, keepdim=True)

        teammate_logits = self.world_model.teammate_logits(latent[:, :-1])
        # Joint actions are repeated once for every focal view of an episode.
        joint_actions = actions[:, :length, :, 0]
        joint_actions = joint_actions[:, None].expand(-1, n_agents, -1, -1)
        joint_actions = joint_actions.reshape(bsz * n_agents, length, n_agents)
        teammate_ce = F.cross_entropy(
            teammate_logits.reshape(-1, self.n_actions),
            joint_actions.reshape(-1), reduction="none"
        ).view(bsz * n_agents, length, n_agents)
        non_focal = th.ones_like(teammate_ce)
        non_focal[
            th.arange(bsz * n_agents, device=obs.device), :, focal
        ] = 0.0
        teammate_loss = (
            (teammate_ce * non_focal).sum(-1, keepdim=True)
            / non_focal.sum(-1, keepdim=True).clamp_min(1.0)
        )

        posterior_next = posterior_logits[:, 1:]
        dynamics_logits = heads["dynamics_logits"]
        target_probs = posterior_next.softmax(-1)
        dynamics_kl = (
            target_probs.detach()
            * (target_probs.detach().clamp_min(1e-8).log() - dynamics_logits.log_softmax(-1))
        ).sum(-1).mean(-1, keepdim=True)
        representation_kl = (
            target_probs
            * (target_probs.clamp_min(1e-8).log() - dynamics_logits.detach().log_softmax(-1))
        ).sum(-1).mean(-1, keepdim=True)
        free_bits = getattr(self.args, "matwm_free_bits", 1.0)
        dynamics_kl = dynamics_kl.clamp_min(free_bits)
        representation_kl = representation_kl.clamp_min(free_bits)

        beta_dyn = getattr(self.args, "matwm_dynamics_weight", 0.5)
        beta_rep = getattr(self.args, "matwm_representation_weight", 0.1)
        total = (
            reconstruction_loss + reward_loss + continuation_loss + teammate_loss
            + beta_dyn * (mask_loss + dynamics_kl)
            + beta_rep * representation_kl
        )
        loss = self._masked_mean(total, focal_valid)

        self.world_optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(
            self.world_model.parameters(),
            getattr(self.args, "matwm_world_grad_clip", 1000.0),
        )
        self.world_optimiser.step()

        return {
            "matwm_world_loss": loss.item(),
            "matwm_reconstruction_loss": self._masked_mean(reconstruction_loss, focal_valid).item(),
            "matwm_reward_loss": self._masked_mean(reward_loss, focal_valid).item(),
            "matwm_continuation_loss": self._masked_mean(continuation_loss, focal_valid).item(),
            "matwm_teammate_loss": self._masked_mean(teammate_loss, focal_valid).item(),
            "matwm_mask_loss": self._masked_mean(mask_loss, focal_valid).item(),
            "matwm_dynamics_kl": self._masked_mean(dynamics_kl, focal_valid).item(),
            "matwm_representation_kl": self._masked_mean(representation_kl, focal_valid).item(),
            "matwm_world_grad_norm": float(grad_norm),
        }

    def _sample_contexts(self, batch):
        context_length = getattr(self.args, "matwm_context_length", 8)
        filled = batch["filled"].squeeze(-1)
        trainable = batch.data.transition_data.get("trainable_agents")
        candidates = []
        for batch_id in range(batch.batch_size):
            valid_length = int(filled[batch_id].sum().item())
            if valid_length < context_length:
                continue
            for agent_id in range(self.n_agents):
                if trainable is not None and not bool(trainable[batch_id, 0, agent_id, 0]):
                    continue
                candidates.append((batch_id, agent_id, valid_length))
        if not candidates:
            return None

        sample_count = self.agent_batch_size
        choice = th.randint(len(candidates), (sample_count,), device=batch.device)
        observations, actions, focals = [], [], []
        for index in choice.tolist():
            batch_id, agent_id, valid_length = candidates[index]
            end = int(th.randint(
                context_length - 1, valid_length, (1,), device=batch.device
            ).item())
            start = end - context_length + 1
            observations.append(batch["obs"][batch_id, start:end + 1, agent_id])
            actions.append(batch["actions"][batch_id, start:end, agent_id])
            focals.append(agent_id)
        return (
            th.stack(observations), th.stack(actions),
            th.tensor(focals, device=batch.device, dtype=th.long),
        )

    def _train_agents(self, batch):
        sampled = self._sample_contexts(batch)
        if sampled is None:
            return {"matwm_actor_loss": 0.0, "matwm_critic_loss": 0.0}
        observations, context_actions, focal = sampled
        horizon = self.imagination_horizon
        gamma = self.args.gamma
        lambda_ = getattr(self.args, "matwm_lambda", 0.95)

        # The learned environment is fixed during policy improvement.
        with th.no_grad():
            latent_history, _ = self.world_model.encode(observations, sample=True)
            hidden = self.world_model.dynamics_sequence(
                latent_history[:, :-1], context_actions, focal
            )[:, -1]
            teammate = self.world_model.teammate_logits(latent_history)[:, -1]
        action_history = context_actions

        log_probs, entropies, values, ema_values = [], [], [], []
        rewards, continuations = [], []
        survival = observations.new_ones(observations.shape[0], 1)
        weights = []

        for _ in range(horizon):
            state = self.policy.build_state(
                latent_history[:, -1], hidden, teammate, focal
            ).detach()
            logits = self.policy.actor_logits(state, focal)
            with th.no_grad():
                available = self.world_model.predicted_availability(
                    latent_history[:, -1]
                )
            logits = logits.masked_fill(~available, -1e9)
            distribution = Categorical(logits=logits)
            action = distribution.sample()

            log_probs.append(distribution.log_prob(action).unsqueeze(-1))
            entropies.append(distribution.entropy().unsqueeze(-1))
            values.append(self.policy.values(state, focal))
            with th.no_grad():
                ema_values.append(self.policy.values(state, focal, ema=True))
            weights.append(survival)

            with th.no_grad():
                action_history = th.cat((action_history, action[:, None, None]), dim=1)
                hidden_sequence = self.world_model.dynamics_sequence(
                    latent_history, action_history, focal
                )
                hidden = hidden_sequence[:, -1]
                heads = self.world_model.prediction_heads(hidden)
                reward = self.world_model.reward_value(heads["reward_logits"])
                continuation = heads["continuation_logits"].sigmoid()
                next_index = Categorical(logits=heads["dynamics_logits"]).sample()
                next_latent = F.one_hot(
                    next_index, self.world_model.n_categories
                ).to(hidden.dtype)
                latent_history = th.cat((latent_history, next_latent[:, None]), dim=1)
                teammate = self.world_model.teammate_logits(latent_history)[:, -1]
                # Keep both histories aligned after the WM truncates its context.
                if latent_history.shape[1] > self.world_model.max_seq_length:
                    latent_history = latent_history[:, -self.world_model.max_seq_length:]
                    action_history = action_history[:, -(self.world_model.max_seq_length - 1):]
            rewards.append(reward)
            continuations.append(continuation)
            survival = survival * continuation

        with th.no_grad():
            final_state = self.policy.build_state(
                latent_history[:, -1], hidden, teammate, focal
            )
            next_return = self.policy.values(final_state, focal)

        returns = []
        for step in reversed(range(horizon)):
            bootstrap = (1.0 - lambda_) * (
                values[step + 1].detach() if step + 1 < horizon else next_return
            ) + lambda_ * next_return
            next_return = rewards[step] + gamma * continuations[step] * bootstrap
            returns.append(next_return)
        returns.reverse()

        log_probs = th.stack(log_probs, 1)
        entropies = th.stack(entropies, 1)
        values = th.stack(values, 1)
        ema_values = th.stack(ema_values, 1)
        returns = th.stack(returns, 1)
        weights = th.stack(weights, 1).detach()

        scale = (th.quantile(returns.detach(), 0.95) - th.quantile(
            returns.detach(), 0.05
        )).clamp_min(1.0)
        advantage = (returns - values).detach() / scale
        entropy_coefficient = getattr(self.args, "matwm_entropy_coef", 3e-4)
        actor_loss = -self._masked_mean(
            advantage * log_probs + entropy_coefficient * entropies, weights
        )
        critic_loss = self._masked_mean(
            (values - returns.detach()).pow(2)
            + (values - ema_values.detach()).pow(2),
            weights,
        )
        loss = actor_loss + critic_loss

        self.agent_optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(
            list(self.policy.actor_parameters()) + list(self.policy.critic_parameters()),
            getattr(self.args, "matwm_agent_grad_clip", 100.0),
        )
        self.agent_optimiser.step()
        self.policy.update_ema(getattr(self.args, "matwm_ema_decay", 0.98))

        return {
            "matwm_actor_loss": actor_loss.item(),
            "matwm_critic_loss": critic_loss.item(),
            "matwm_return_mean": returns.mean().item(),
            "matwm_imagination_continue": th.stack(continuations, 1).mean().item(),
            "matwm_agent_grad_norm": float(grad_norm),
        }

    def cuda(self):
        self.mac.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        th.save(self.world_optimiser.state_dict(), os.path.join(path, "world_opt.th"))
        th.save(self.agent_optimiser.state_dict(), os.path.join(path, "agent_opt.th"))

    def load_models(self, path):
        self.mac.load_models(path)
        self.world_optimiser.load_state_dict(
            th.load(os.path.join(path, "world_opt.th"), map_location="cpu")
        )
        self.agent_optimiser.load_state_dict(
            th.load(os.path.join(path, "agent_opt.th"), map_location="cpu")
        )
