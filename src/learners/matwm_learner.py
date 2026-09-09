"""Learner for the Multi-Agent Transformer World Model (MATWM)."""

import os

import torch as th
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.optim import Adam
from components.standarize_stream import RunningMeanStd


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
        self.separate_ppo_optimisers = getattr(
            args, "matwm_ppo_separate_critic", False
        )
        if self.separate_ppo_optimisers:
            self.actor_optimiser = Adam(
                self.policy.actor_parameters(),
                lr=getattr(args, "matwm_agent_lr", 3e-4),
                eps=getattr(args, "optim_eps", 1e-5),
            )
            self.critic_optimiser = Adam(
                self.policy.critic_parameters(),
                lr=getattr(args, "matwm_ppo_critic_lr", 1e-3),
                eps=getattr(args, "optim_eps", 1e-5),
            )
        self.last_log_t = -getattr(args, "learner_log_interval", 10000) - 1
        self.train_calls = 0
        self.last_agent_stats = None
        self.last_world_stats = None
        device = "cuda" if getattr(args, "use_cuda", False) else "cpu"
        self.real_reward_normalizer = (
            RunningMeanStd(shape=(1,), device=device)
            if getattr(args, "matwm_ppo_standardise_rewards", False)
            else None
        )
        self.real_return_normalizer = (
            RunningMeanStd(shape=(1,), device=device)
            if getattr(args, "matwm_ppo_standardise_returns", False)
            else None
        )

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
        self.train_calls += 1
        world_stats = self._train_world_model(batch)
        self.last_world_stats = world_stats
        use_real_policy = getattr(self.args, "matwm_real_reward_policy", False)
        real_policy_interval = getattr(
            self.args, "matwm_real_policy_update_interval", 1
        )
        should_train_policy = (
            not use_real_policy
            or self.train_calls % real_policy_interval == 0
        )
        if (
            t_env >= getattr(self.args, "matwm_prefill_steps", 1000)
            and should_train_policy
        ):
            agent_stats = (
                self._train_agents_real(batch)
                if use_real_policy else self._train_agents(batch)
            )
            self.last_agent_stats = agent_stats
        elif self.last_agent_stats is not None:
            agent_stats = self.last_agent_stats
        else:
            agent_stats = {
                "matwm_actor_loss": 0.0,
                "matwm_critic_loss": 0.0,
                "matwm_agent_grad_norm": 0.0,
                "matwm_return_mean": 0.0,
                "matwm_imagination_continue": 0.0,
            }

        if t_env - self.last_log_t >= self.args.learner_log_interval:
            for key, value in {**world_stats, **agent_stats}.items():
                self.logger.log_stat(key, value, t_env)
            self.last_log_t = t_env

    def train_world_only(self, batch, t_env, episode_num):
        self.train_calls += 1
        self.last_world_stats = self._train_world_model(batch)

    def train_real_ppo(self, batch, t_env, episode_num):
        if t_env < getattr(self.args, "matwm_prefill_steps", 1000):
            return

        batch_size = batch.batch_size
        transitions = batch["reward"].shape[1] - 1
        agent_indices = list(range(self.n_agents))
        states = []
        with th.no_grad():
            for t in range(transitions + 1):
                state, _, _, _, _ = self.policy.real_state(
                    batch, t, agent_indices, test_mode=False
                )
                states.append(
                    state.view(batch_size, self.n_agents, -1)
                )
        states = th.stack(states, dim=1)

        actions = batch["actions"][:, :transitions, :, 0].long()
        available = batch["avail_actions"][:, :transitions].bool()
        rewards = batch["reward"][:, :transitions, 0]
        if self.real_reward_normalizer is not None:
            flat_rewards = rewards.reshape(-1, 1)
            self.real_reward_normalizer.update(flat_rewards)
            rewards = (
                rewards - self.real_reward_normalizer.mean.squeeze(-1)
            ) / th.sqrt(
                self.real_reward_normalizer.var.squeeze(-1)
            ).clamp_min(1e-4)
        terminated = batch["terminated"][:, :transitions, 0].float()
        filled = batch["filled"][:, :transitions, 0].float()
        trainable = batch.data.transition_data.get("trainable_agents")
        if trainable is None:
            trainable_mask = filled[:, :, None].expand(-1, -1, self.n_agents)
        else:
            trainable_mask = (
                filled[:, :, None] * trainable[:, :transitions, :, 0].float()
            )

        focal = th.arange(self.n_agents, device=batch.device).repeat(batch_size)
        with th.no_grad():
            old_logits = []
            old_values_normalized = []
            for t in range(transitions + 1):
                flat_state = states[:, t].reshape(batch_size * self.n_agents, -1)
                old_values_normalized.append(
                    self.policy.values(flat_state, focal).view(
                        batch_size, self.n_agents
                    )
                )
                if t < transitions:
                    logits = self.policy.actor_logits(flat_state, focal).view(
                        batch_size, self.n_agents, self.n_actions
                    )
                    logits = logits.masked_fill(~available[:, t], -1e9)
                    old_logits.append(logits)
            old_values_normalized = th.stack(old_values_normalized, dim=1)
            if self.real_return_normalizer is None:
                old_values = old_values_normalized
            else:
                return_scale = th.sqrt(
                    self.real_return_normalizer.var.squeeze(-1)
                ).clamp_min(1e-4)
                old_values = (
                    old_values_normalized * return_scale
                    + self.real_return_normalizer.mean.squeeze(-1)
                )
            old_distribution = Categorical(logits=th.stack(old_logits, dim=1))
            old_log_prob = old_distribution.log_prob(actions)

            advantages = th.zeros_like(old_values[:, :-1])
            gae = th.zeros_like(old_values[:, 0])
            gamma = self.args.gamma
            gae_lambda = getattr(self.args, "matwm_ppo_gae_lambda", 0.95)
            for t in reversed(range(transitions)):
                continues = 1.0 - terminated[:, t]
                delta = (
                    rewards[:, t, None]
                    + gamma * continues[:, None] * old_values[:, t + 1]
                    - old_values[:, t]
                )
                gae = delta + gamma * gae_lambda * continues[:, None] * gae
                advantages[:, t] = gae
            returns = advantages + old_values[:, :-1]
            valid_advantages = advantages[trainable_mask.bool()]
            advantages = (
                advantages - valid_advantages.mean()
            ) / valid_advantages.std(unbiased=False).clamp_min(1e-5)

            if self.real_return_normalizer is None:
                critic_returns = returns
            else:
                self.real_return_normalizer.update(
                    returns[trainable_mask.bool()].reshape(-1, 1)
                )
                critic_returns = (
                    returns - self.real_return_normalizer.mean.squeeze(-1)
                ) / th.sqrt(
                    self.real_return_normalizer.var.squeeze(-1)
                ).clamp_min(1e-4)

        flat_states = states[:, :-1].reshape(-1, states.shape[-1])
        flat_actions = actions.reshape(-1)
        flat_available = available.reshape(-1, self.n_actions)
        flat_old_log_prob = old_log_prob.reshape(-1)
        flat_old_values = old_values_normalized[:, :-1].reshape(-1)
        flat_advantages = advantages.reshape(-1)
        flat_returns = critic_returns.reshape(-1)
        valid_indices = trainable_mask.reshape(-1).nonzero().squeeze(-1)

        epochs = getattr(self.args, "matwm_ppo_epochs", 4)
        minibatches = getattr(self.args, "matwm_ppo_minibatches", 4)
        clip = getattr(self.args, "matwm_ppo_clip", 0.1)
        value_coefficient = getattr(self.args, "matwm_ppo_value_coef", 0.5)
        entropy_coefficient = getattr(self.args, "matwm_entropy_coef", 3e-4)
        actor_losses, critic_losses, ratios, entropies, grad_norms = [], [], [], [], []
        approximate_kls, clip_fractions = [], []

        for _ in range(epochs):
            shuffled = valid_indices[th.randperm(valid_indices.numel(), device=batch.device)]
            for indices in shuffled.chunk(minibatches):
                state = flat_states[indices]
                focal_ids = (indices % self.n_agents).long()
                logits = self.policy.actor_logits(state, focal_ids)
                logits = logits.masked_fill(~flat_available[indices], -1e9)
                distribution = Categorical(logits=logits)
                log_prob = distribution.log_prob(flat_actions[indices])
                ratio = (log_prob - flat_old_log_prob[indices]).exp()
                advantage = flat_advantages[indices]
                surrogate = th.minimum(
                    ratio * advantage,
                    ratio.clamp(1.0 - clip, 1.0 + clip) * advantage,
                )
                actor_loss = -(
                    surrogate
                    + entropy_coefficient * distribution.entropy()
                ).mean()

                if self.separate_ppo_optimisers:
                    self.actor_optimiser.zero_grad()
                    actor_loss.backward()
                    grad_norm = th.nn.utils.clip_grad_norm_(
                        self.policy.actor_parameters(),
                        getattr(self.args, "matwm_agent_grad_clip", 100.0),
                    )
                    self.actor_optimiser.step()
                else:
                    values = self.policy.values(state, focal_ids).squeeze(-1)
                    clipped_values = flat_old_values[indices] + (
                        values - flat_old_values[indices]
                    ).clamp(-clip, clip)
                    value_loss = th.maximum(
                        (values - flat_returns[indices]).pow(2),
                        (clipped_values - flat_returns[indices]).pow(2),
                    ).mean()
                    self.agent_optimiser.zero_grad()
                    (actor_loss + value_coefficient * value_loss).backward()
                    grad_norm = th.nn.utils.clip_grad_norm_(
                        list(self.policy.actor_parameters())
                        + list(self.policy.critic_parameters()),
                        getattr(self.args, "matwm_agent_grad_clip", 100.0),
                    )
                    self.agent_optimiser.step()
                    critic_losses.append(value_loss.item())
                actor_losses.append(actor_loss.item())
                ratios.append(ratio.mean().item())
                entropies.append(distribution.entropy().mean().item())
                grad_norms.append(float(grad_norm))
                approximate_kls.append(
                    (flat_old_log_prob[indices] - log_prob).mean().item()
                )
                clip_fractions.append(
                    ((ratio - 1.0).abs() > clip).float().mean().item()
                )

        if self.separate_ppo_optimisers:
            critic_epochs = getattr(self.args, "matwm_ppo_critic_epochs", epochs)
            for _ in range(critic_epochs):
                shuffled = valid_indices[
                    th.randperm(valid_indices.numel(), device=batch.device)
                ]
                for indices in shuffled.chunk(minibatches):
                    state = flat_states[indices]
                    focal_ids = (indices % self.n_agents).long()
                    values = self.policy.values(state, focal_ids).squeeze(-1)
                    clipped_values = flat_old_values[indices] + (
                        values - flat_old_values[indices]
                    ).clamp(-clip, clip)
                    value_loss = th.maximum(
                        (values - flat_returns[indices]).pow(2),
                        (clipped_values - flat_returns[indices]).pow(2),
                    ).mean()
                    self.critic_optimiser.zero_grad()
                    value_loss.backward()
                    th.nn.utils.clip_grad_norm_(
                        self.policy.critic_parameters(),
                        getattr(self.args, "matwm_agent_grad_clip", 100.0),
                    )
                    self.critic_optimiser.step()
                    critic_losses.append(value_loss.item())

        self.policy.update_ema(getattr(self.args, "matwm_ema_decay", 0.98))
        agent_stats = {
            "matwm_actor_loss": sum(actor_losses) / len(actor_losses),
            "matwm_critic_loss": sum(critic_losses) / len(critic_losses),
            "matwm_real_critic_loss": sum(critic_losses) / len(critic_losses),
            "matwm_return_mean": returns[trainable_mask.bool()].mean().item(),
            "matwm_imagination_continue": 1.0,
            "matwm_agent_grad_norm": sum(grad_norms) / len(grad_norms),
            "matwm_ppo_ratio": sum(ratios) / len(ratios),
            "matwm_ppo_entropy": sum(entropies) / len(entropies),
            "matwm_ppo_approx_kl": sum(approximate_kls) / len(approximate_kls),
            "matwm_ppo_clip_fraction": sum(clip_fractions) / len(clip_fractions),
            "matwm_real_advantage_std": valid_advantages.std(unbiased=False).item(),
            "matwm_ppo_reward_mean": rewards.mean().item(),
            "matwm_ppo_reward_std": rewards.std(unbiased=False).item(),
            "matwm_ppo_value_target_mean": critic_returns[
                trainable_mask.bool()
            ].mean().item(),
            "matwm_ppo_value_target_std": critic_returns[
                trainable_mask.bool()
            ].std(unbiased=False).item(),
        }
        self.last_agent_stats = agent_stats
        if t_env - self.last_log_t >= self.args.learner_log_interval:
            stats = {**(self.last_world_stats or {}), **agent_stats}
            for key, value in stats.items():
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
        dynamics_actions = focal_actions[:, :-1]
        if getattr(self.args, "matwm_joint_action_dynamics", False):
            joint = actions[:, :length, :, 0]
            dynamics_actions = joint[:, None].expand(-1, n_agents, -1, -1)
            dynamics_actions = dynamics_actions.reshape(
                bsz * n_agents, length, n_agents
            )
            predicted_ratio = getattr(
                self.args, "matwm_predicted_teammate_ratio", 0.0
            )
            if predicted_ratio > 0:
                predicted = self.policy.predicted_joint_actions(
                    latent[:, :-1], focal, focal_actions[:, :-1]
                )
                if predicted.dim() == 4:
                    dynamics_actions = F.one_hot(
                        dynamics_actions.long(), self.n_actions
                    ).to(predicted.dtype)
                replace = th.rand(
                    dynamics_actions.shape[:2], device=obs.device
                ) < predicted_ratio
                while replace.dim() < dynamics_actions.dim():
                    replace = replace.unsqueeze(-1)
                dynamics_actions = th.where(
                    replace, predicted.detach(), dynamics_actions
                )
        hidden = self.world_model.dynamics_sequence(
            latent[:, :-1], dynamics_actions, focal
        )
        if hasattr(self.world_model, "autoregressive_dynamics_logits"):
            heads = self.world_model.prediction_heads(
                hidden, latent[:, :-1], latent[:, 1:].detach()
            )
        else:
            heads = self.world_model.prediction_heads(hidden, latent[:, :-1])

        reconstruction_loss = (reconstruction[:, :-1] - focal_obs[:, :-1]).pow(2).mean(-1, keepdim=True)

        reward_target = rewards[:, :length, None].expand(-1, -1, n_agents, -1)
        reward_target = reward_target.permute(0, 2, 1, 3).reshape(
            bsz * n_agents, length, 1
        )
        if self.world_model.use_reward_model:
            if self.world_model.reward_regression:
                reward_losses = [
                    F.smooth_l1_loss(
                        logits, reward_target, reduction="none"
                    )
                    for logits in heads["reward_ensemble_logits"]
                ]
            else:
                two_hot = self.world_model.two_hot_reward(reward_target)
                reward_losses = [
                    -(two_hot * logits.log_softmax(-1)).sum(-1, keepdim=True)
                    for logits in heads["reward_ensemble_logits"]
                ]
            reward_loss = th.stack(reward_losses).mean(0)
            with th.no_grad():
                predicted_rewards = th.stack([
                    self.world_model.reward_value(logits)
                    for logits in heads["reward_ensemble_logits"]
                ]).mean(0)
                reward_error = predicted_rewards - reward_target
                reward_mae = self._masked_mean(reward_error.abs(), focal_valid)
                reward_bias = self._masked_mean(reward_error, focal_valid)

                nonzero = (reward_target.abs() > getattr(
                    self.args, "matwm_reward_zero_epsilon", 1e-6
                )).to(focal_valid.dtype) * focal_valid
                nonzero_mae = self._masked_mean(reward_error.abs(), nonzero)

                zero = (1.0 - nonzero) * focal_valid
                false_positive = (predicted_rewards.abs() > getattr(
                    self.args, "matwm_reward_prediction_threshold", 0.1
                )).to(focal_valid.dtype)
                zero_false_positive_rate = self._masked_mean(false_positive, zero)

                event_threshold = getattr(self.args, "matwm_reward_event_threshold", 1.0)
                events = (reward_target.abs() >= event_threshold).to(focal_valid.dtype) * focal_valid
                detected = (predicted_rewards.abs() >= event_threshold).to(focal_valid.dtype)
                event_recall = self._masked_mean(detected, events)

                if length >= 3:
                    gamma = self.args.gamma
                    predicted_three = (
                        predicted_rewards[:, :-2]
                        + gamma * predicted_rewards[:, 1:-1]
                        + gamma ** 2 * predicted_rewards[:, 2:]
                    )
                    target_three = (
                        reward_target[:, :-2]
                        + gamma * reward_target[:, 1:-1]
                        + gamma ** 2 * reward_target[:, 2:]
                    )
                    valid_three = (
                        focal_valid[:, :-2] * focal_valid[:, 1:-1] * focal_valid[:, 2:]
                    )
                    three_step_mae = self._masked_mean(
                        (predicted_three - target_three).abs(), valid_three
                    )
                else:
                    three_step_mae = reward_mae.new_zeros(())
        else:
            reward_loss = reconstruction_loss.new_zeros(reconstruction_loss.shape)
            reward_mae = reward_bias = nonzero_mae = reward_loss.new_zeros(())
            zero_false_positive_rate = event_recall = three_step_mae = reward_mae

        continue_target = (1.0 - terminated[:, :length])[:, None].expand(
            -1, n_agents, -1, -1
        ).reshape(bsz * n_agents, length, 1)
        continuation_loss = F.binary_cross_entropy_with_logits(
            heads["continuation_logits"], continue_target, reduction="none"
        )
        if getattr(self.args, "marie_original_procedure", False):
            # Upstream predicts availability at t+1 from the (z_t, a_t)
            # Transformer state.
            mask_loss = F.binary_cross_entropy_with_logits(
                heads["availability_logits"], focal_available[:, 1:],
                reduction="none",
            ).mean(-1, keepdim=True)
        else:
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
            "matwm_reward_mae": reward_mae.item(),
            "matwm_reward_bias": reward_bias.item(),
            "matwm_reward_nonzero_mae": nonzero_mae.item(),
            "matwm_reward_zero_false_positive_rate": zero_false_positive_rate.item(),
            "matwm_reward_event_recall": event_recall.item(),
            "matwm_reward_three_step_teacher_forced_mae": three_step_mae.item(),
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
        if getattr(self.args, "marie_original_procedure", False):
            # The upstream episode dataset already returns exactly one
            # endpoint-aligned, left-padded context per requested sample. Do
            # not resample endpoints or discard padded early-episode contexts.
            observations = batch["obs"][:, -context_length:].permute(
                0, 2, 1, 3
            ).reshape(
                batch.batch_size * self.n_agents, context_length, -1
            )
            actions = batch["actions"][:, -context_length:-1].permute(
                0, 2, 1, 3
            ).reshape(
                batch.batch_size * self.n_agents, context_length - 1, 1
            )
            joint_actions = batch["actions"][
                :, -context_length:-1, :, 0
            ][:, None].expand(-1, self.n_agents, -1, -1).reshape(
                batch.batch_size * self.n_agents,
                context_length - 1, self.n_agents,
            )
            current_actions = batch["actions"][:, -1, :, 0].reshape(-1)
            current_available = batch["avail_actions"][:, -1].reshape(
                batch.batch_size * self.n_agents, self.n_actions
            )
            focal = th.arange(
                self.n_agents, device=batch.device
            ).repeat(batch.batch_size)
            real_returns = observations.new_zeros(
                batch.batch_size * self.n_agents, 1
            )
            return (
                observations, actions, joint_actions, current_actions,
                current_available, real_returns, focal,
            )
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

        # MARIE's Perceiver must see complete teams.  Sample a common episode
        # and temporal window and then emit every focal agent contiguously so
        # dynamics_sequence can aggregate them as [batch, agents, time, dim].
        if getattr(self.args, "marie_grouped_contexts", False):
            team_candidates = []
            for batch_id in range(batch.batch_size):
                valid_length = int(filled[batch_id].sum().item())
                if valid_length < context_length:
                    continue
                if trainable is not None and not bool(
                    trainable[batch_id, 0, :, 0].all()
                ):
                    continue
                team_candidates.append((batch_id, valid_length))
            if not team_candidates:
                return None
            team_count = max(1, self.agent_batch_size // self.n_agents)
            choice = th.randint(
                len(team_candidates), (team_count,), device=batch.device
            )
            selected = []
            for index in choice.tolist():
                batch_id, valid_length = team_candidates[index]
                end = int(th.randint(
                    context_length - 1, valid_length, (1,),
                    device=batch.device,
                ).item())
                selected.extend(
                    (batch_id, agent_id, valid_length, end)
                    for agent_id in range(self.n_agents)
                )
        else:
            sample_count = self.agent_batch_size
            choice = th.randint(
                len(candidates), (sample_count,), device=batch.device
            )
            selected = []
            for index in choice.tolist():
                batch_id, agent_id, valid_length = candidates[index]
                end = int(th.randint(
                    context_length - 1, valid_length, (1,),
                    device=batch.device,
                ).item())
                selected.append((batch_id, agent_id, valid_length, end))

        observations, actions, joint_actions = [], [], []
        current_actions, current_available = [], []
        real_returns, focals = [], []
        for batch_id, agent_id, valid_length, end in selected:
            start = end - context_length + 1
            observations.append(batch["obs"][batch_id, start:end + 1, agent_id])
            actions.append(batch["actions"][batch_id, start:end, agent_id])
            joint_actions.append(batch["actions"][batch_id, start:end, :, 0])
            current_actions.append(batch["actions"][batch_id, end, agent_id, 0])
            current_available.append(
                batch["avail_actions"][batch_id, end, agent_id]
            )
            future_rewards = batch["reward"][batch_id, end:valid_length, 0]
            discounts = self.args.gamma ** th.arange(
                future_rewards.shape[0], device=batch.device
            )
            real_returns.append((future_rewards * discounts).sum())
            focals.append(agent_id)
        return (
            th.stack(observations), th.stack(actions), th.stack(joint_actions),
            th.stack(current_actions), th.stack(current_available),
            th.stack(real_returns).unsqueeze(-1),
            th.tensor(focals, device=batch.device, dtype=th.long),
        )

    def _train_agents_real(self, batch):
        """Train a replay policy and value function using environment returns."""
        sampled = self._sample_contexts(batch)
        if sampled is None:
            return {"matwm_actor_loss": 0.0, "matwm_critic_loss": 0.0}
        (
            observations, context_actions, context_joint_actions,
            actions, available, real_returns, focal,
        ) = sampled

        with th.no_grad():
            latent_history, _ = self.world_model.encode(observations, sample=True)
            dynamics_actions = context_actions
            if getattr(self.args, "matwm_joint_action_dynamics", False):
                dynamics_actions = context_joint_actions
            hidden = self.world_model.dynamics_sequence(
                latent_history[:, :-1], dynamics_actions, focal
            )[:, -1]
            teammate = self.world_model.teammate_logits(latent_history)[:, -1]
            state = self.policy.build_state(
                latent_history[:, -1], hidden, teammate, focal
            )

        logits = self.policy.actor_logits(state, focal)
        logits = logits.masked_fill(~available.bool(), -1e9)
        distribution = Categorical(logits=logits)
        log_prob = distribution.log_prob(actions).unsqueeze(-1)
        entropy = distribution.entropy().unsqueeze(-1)
        values = self.policy.values(state, focal)

        advantage = (real_returns - values.detach())
        normalized_advantage = (
            advantage - advantage.mean()
        ) / advantage.std(unbiased=False).clamp_min(1e-4)
        temperature = getattr(self.args, "matwm_awr_temperature", 1.0)
        max_weight = getattr(self.args, "matwm_awr_max_weight", 20.0)
        awr_weight = (normalized_advantage / temperature).exp().clamp_max(
            max_weight
        )
        awr_weight = awr_weight / awr_weight.mean().clamp_min(1e-6)

        entropy_coefficient = getattr(self.args, "matwm_entropy_coef", 3e-4)
        actor_loss = -(
            awr_weight.detach() * log_prob
            + entropy_coefficient * entropy
        ).mean()
        critic_loss = F.smooth_l1_loss(values, real_returns.detach())
        loss = actor_loss + critic_loss

        self.agent_optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(
            list(self.policy.actor_parameters())
            + list(self.policy.critic_parameters()),
            getattr(self.args, "matwm_agent_grad_clip", 100.0),
        )
        self.agent_optimiser.step()
        self.policy.update_ema(getattr(self.args, "matwm_ema_decay", 0.98))

        return {
            "matwm_actor_loss": actor_loss.item(),
            "matwm_critic_loss": critic_loss.item(),
            "matwm_real_critic_loss": critic_loss.item(),
            "matwm_return_mean": real_returns.mean().item(),
            "matwm_imagination_continue": 1.0,
            "matwm_agent_grad_norm": float(grad_norm),
            "matwm_real_advantage_std": advantage.std(unbiased=False).item(),
            "matwm_awr_weight_mean": awr_weight.mean().item(),
            "matwm_awr_weight_max": awr_weight.max().item(),
        }

    def _train_agents(self, batch):
        sampled = self._sample_contexts(batch)
        if sampled is None:
            return {"matwm_actor_loss": 0.0, "matwm_critic_loss": 0.0}
        (
            observations, context_actions, context_joint_actions,
            _, current_available, real_returns, focal,
        ) = sampled
        horizon = self.imagination_horizon
        gamma = self.args.gamma
        lambda_ = getattr(self.args, "matwm_lambda", 0.95)
        original_marie = getattr(
            self.args, "marie_original_procedure", False
        )

        # The learned environment is fixed during policy improvement.
        with th.no_grad():
            encoded_history, _ = self.world_model.encode(observations, sample=True)
            use_dynamics_cache = hasattr(
                self.world_model, "init_dynamics_cache"
            )
            if original_marie:
                # Upstream reset_from_initial_observations starts at the final
                # observation. Earlier frames only form the policy stack.
                latent_history = encoded_history[:, -1:]
                hidden = observations.new_zeros(
                    observations.shape[0], self.world_model.hidden_dim
                )
                dynamics_cache = None
                action_history = context_actions[:, :0]
            else:
                latent_history = encoded_history
                dynamics_actions = context_actions
                if getattr(self.args, "matwm_joint_action_dynamics", False):
                    dynamics_actions = context_joint_actions
                    if getattr(self.args, "matwm_soft_teammate_actions", False):
                        dynamics_actions = F.one_hot(
                            dynamics_actions.long(), self.n_actions
                        ).to(observations.dtype)
                if use_dynamics_cache:
                    hidden_sequence, dynamics_cache = (
                        self.world_model.init_dynamics_cache(
                            latent_history[:, :-1], dynamics_actions, focal
                        )
                    )
                else:
                    hidden_sequence = self.world_model.dynamics_sequence(
                        latent_history[:, :-1], dynamics_actions, focal
                    )
                    dynamics_cache = None
                hidden = hidden_sequence[:, -1]
                action_history = dynamics_actions
            teammate = self.world_model.teammate_logits(latent_history)[:, -1]
        imagined_observations = observations

        log_probs, entropies, values, ema_values = [], [], [], []
        rollout_states, rollout_actions, rollout_available = [], [], []
        rewards, continuations = [], []
        imagined_available = current_available.bool()
        survival = observations.new_ones(observations.shape[0], 1)
        weights = []

        for _ in range(horizon):
            state = self.policy.build_state(
                latent_history[:, -1], hidden, teammate, focal,
                imagined_observations
                if getattr(self.args, "marie_stack_obs", 0) else None,
            ).detach()
            logits = self.policy.actor_logits(state, focal)
            with th.no_grad():
                available = (
                    imagined_available
                    if original_marie else
                    self.world_model.predicted_availability(
                        latent_history[:, -1]
                    )
                )
            logits = logits.masked_fill(~available, -1e9)
            distribution = Categorical(logits=logits)
            action = distribution.sample()

            rollout_states.append(state)
            rollout_actions.append(action)
            rollout_available.append(available)
            log_probs.append(distribution.log_prob(action).unsqueeze(-1))
            entropies.append(distribution.entropy().unsqueeze(-1))
            values.append(self.policy.values(state, focal))
            with th.no_grad():
                ema_values.append(self.policy.values(state, focal, ema=True))
            if not original_marie:
                weights.append(survival)
            with th.no_grad():
                next_actions = action[:, None, None]
                if getattr(self.args, "matwm_joint_action_dynamics", False):
                    next_actions = self.policy.predicted_joint_actions(
                        latent_history[:, -1:], focal, next_actions
                    )
                action_history = th.cat((action_history, next_actions), dim=1)
                if use_dynamics_cache and dynamics_cache is None:
                    hidden_sequence, dynamics_cache = (
                        self.world_model.init_dynamics_cache(
                            latent_history[:, -1:], next_actions, focal
                        )
                    )
                elif use_dynamics_cache and original_marie:
                    hidden, dynamics_cache = (
                        self.world_model.append_action_cache(
                            latent_history[:, -1:], next_actions, focal,
                            dynamics_cache,
                        )
                    )
                    hidden_sequence = hidden[:, None]
                elif use_dynamics_cache:
                    hidden_sequence, dynamics_cache = (
                        self.world_model.append_dynamics_cache(
                            latent_history[:, -1:], next_actions, focal,
                            dynamics_cache,
                        )
                    )
                else:
                    hidden_sequence = self.world_model.dynamics_sequence(
                        latent_history, action_history, focal
                    )
                hidden = hidden_sequence[:, -1]
                heads = self.world_model.prediction_heads(
                    hidden, latent_history[:, -1]
                )
                reward_predictions = th.stack([
                    self.world_model.reward_value(logits)
                    for logits in heads["reward_ensemble_logits"]
                ], 0)
                reward = reward_predictions.mean(0)
                uncertainty = reward_predictions.std(0, unbiased=False)
                reward = reward - getattr(
                    self.args, "matwm_uncertainty_penalty", 0.0
                ) * uncertainty
                continuation = heads["continuation_logits"].sigmoid()
                if original_marie:
                    imagined_available = (
                        self.world_model.predicted_availability_from_hidden(
                            hidden
                        )
                    )
                if (
                    original_marie
                    and dynamics_cache is not None
                    and hasattr(self.world_model, "sample_next_latent_cached")
                ):
                    next_latent, hidden, dynamics_cache = (
                        self.world_model.sample_next_latent_cached(
                            hidden, dynamics_cache
                        )
                    )
                elif hasattr(self.world_model, "sample_next_latent"):
                    next_latent = self.world_model.sample_next_latent(hidden)
                else:
                    next_index = Categorical(
                        logits=heads["dynamics_logits"]
                    ).sample()
                    next_latent = F.one_hot(
                        next_index, self.world_model.n_categories
                    ).to(hidden.dtype)
                latent_history = th.cat((latent_history, next_latent[:, None]), dim=1)
                if getattr(self.args, "marie_stack_obs", 0):
                    next_observation = self.world_model.decode(next_latent)
                    imagined_observations = th.cat(
                        (imagined_observations, next_observation[:, None]), dim=1
                    )
                    if imagined_observations.shape[1] > self.world_model.max_seq_length:
                        imagined_observations = imagined_observations[
                            :, -self.world_model.max_seq_length:
                        ]
                teammate = self.world_model.teammate_logits(latent_history)[:, -1]
                # Keep both histories aligned after the WM truncates its context.
                if latent_history.shape[1] > self.world_model.max_seq_length:
                    latent_history = latent_history[:, -self.world_model.max_seq_length:]
                    action_history = action_history[:, -(self.world_model.max_seq_length - 1):]
            rewards.append(reward)
            continuations.append(continuation)
            if not original_marie:
                survival = survival * continuation

        with th.no_grad():
            final_state = self.policy.build_state(
                latent_history[:, -1], hidden, teammate, focal,
                imagined_observations
                if getattr(self.args, "marie_stack_obs", 0) else None,
            )
            use_ema_bootstrap = getattr(
                self.args, "matwm_ema_bootstrap", True
            )
            next_return = self.policy.values(
                final_state, focal, ema=use_ema_bootstrap
            )

        returns = []
        for step in reversed(range(horizon)):
            if step + 1 < horizon:
                bootstrap_value = (
                    ema_values[step + 1]
                    if use_ema_bootstrap else values[step + 1].detach()
                )
            else:
                bootstrap_value = next_return
            bootstrap = (1.0 - lambda_) * (
                bootstrap_value
            ) + lambda_ * next_return
            next_return = rewards[step] + gamma * continuations[step] * bootstrap
            returns.append(next_return)
        returns.reverse()

        log_probs = th.stack(log_probs, 1)
        entropies = th.stack(entropies, 1)
        values = th.stack(values, 1)
        ema_values = th.stack(ema_values, 1)
        returns = th.stack(returns, 1)
        if original_marie:
            advantage = (returns - values).detach()
            advantage = (advantage - advantage.mean()) / (
                advantage.std(unbiased=False) + 1e-4
            )
        else:
            weights = th.stack(weights, 1).detach()
            scale = (th.quantile(returns.detach(), 0.95) - th.quantile(
                returns.detach(), 0.05
            )).clamp_min(1.0)
            advantage = (returns - values).detach() / scale
        entropy_coefficient = getattr(self.args, "matwm_entropy_coef", 3e-4)
        if getattr(self.args, "marie_imagined_ppo", False):
            states = th.stack(rollout_states, 1).detach()
            actions = th.stack(rollout_actions, 1).detach()
            available = th.stack(rollout_available, 1).detach()
            old_log_probs = log_probs.detach().squeeze(-1)
            old_values = values.detach().squeeze(-1)
            fixed_returns = returns.detach().squeeze(-1)
            fixed_advantage = advantage.detach().squeeze(-1)
            if original_marie:
                # Preserve complete [agent 0..N-1] teams in every flattened
                # PPO slice so AugmentedCritic attention is active.
                team_count = states.shape[0] // self.n_agents
                def team_time(value):
                    trailing = value.shape[2:]
                    return value.view(
                        team_count, self.n_agents, horizon, *trailing
                    ).transpose(1, 2).reshape(
                        team_count * horizon * self.n_agents, *trailing
                    )
                flat_states = team_time(states)
                flat_actions = team_time(actions).reshape(-1)
                flat_available = team_time(available).reshape(
                    -1, self.n_actions
                )
                flat_old_log_probs = team_time(old_log_probs).reshape(-1)
                flat_old_values = team_time(old_values).reshape(-1)
                flat_returns = team_time(fixed_returns).reshape(-1)
                flat_advantage = team_time(fixed_advantage).reshape(-1)
                flat_focal = th.arange(
                    self.n_agents, device=focal.device
                ).repeat(team_count * horizon)
            else:
                flat_states = states.reshape(-1, states.shape[-1])
                flat_actions = actions.reshape(-1)
                flat_available = available.reshape(-1, self.n_actions)
                flat_old_log_probs = old_log_probs.reshape(-1)
                flat_old_values = old_values.reshape(-1)
                flat_returns = fixed_returns.reshape(-1)
                flat_advantage = fixed_advantage.reshape(-1)
                flat_focal = focal[:, None].expand(-1, horizon).reshape(-1)
            clip = getattr(self.args, "marie_ppo_clip", 0.2)
            value_coefficient = getattr(self.args, "marie_value_coef", 0.5)
            epochs = getattr(self.args, "marie_ppo_epochs", 5)
            actor_losses, critic_losses, grad_norms = [], [], []
            for _ in range(epochs):
                if original_marie:
                    # Upstream shuffles the leading rollout dimension and
                    # processes at most 2,000 agent samples per minibatch.
                    # Shuffle team-time units here so centralized attention
                    # always receives complete teams.
                    unit_count = flat_states.shape[0] // self.n_agents
                    unit_order = th.randperm(
                        unit_count, device=flat_states.device
                    )
                    units_per_minibatch = max(1, 2000 // self.n_agents)
                    minibatches = [
                        (unit_order[start:start + units_per_minibatch, None]
                         * self.n_agents
                         + th.arange(
                             self.n_agents, device=flat_states.device
                         )[None]).reshape(-1)
                        for start in range(0, unit_count, units_per_minibatch)
                    ]
                else:
                    minibatches = [th.arange(
                        flat_states.shape[0], device=flat_states.device
                    )]
                for idx in minibatches:
                    logits = self.policy.actor_logits(
                        flat_states[idx], flat_focal[idx]
                    )
                    logits = logits.masked_fill(~flat_available[idx], -1e9)
                    distribution = Categorical(logits=logits)
                    new_log_prob = distribution.log_prob(flat_actions[idx])
                    ratio = (new_log_prob - flat_old_log_probs[idx]).exp()
                    surrogate = th.minimum(
                        ratio * flat_advantage[idx],
                        ratio.clamp(1.0 - clip, 1.0 + clip)
                        * flat_advantage[idx],
                    )
                    actor_loss = -(
                        surrogate
                        + entropy_coefficient * distribution.entropy()
                    ).mean()
                    new_values = self.policy.values(
                        flat_states[idx], flat_focal[idx]
                    ).squeeze(-1)
                    # Original StarCraft path uses an unclipped MSE/2 critic
                    # loss.
                    critic_loss = 0.5 * (
                        new_values - flat_returns[idx]
                    ).pow(2).mean()
                    if self.separate_ppo_optimisers:
                        self.actor_optimiser.zero_grad()
                        actor_loss.backward()
                        actor_grad_norm = th.nn.utils.clip_grad_norm_(
                            self.policy.actor_parameters(),
                            getattr(
                                self.args, "matwm_agent_grad_clip", 100.0
                            ),
                        )
                        self.actor_optimiser.step()
                        self.critic_optimiser.zero_grad()
                        (value_coefficient * critic_loss).backward()
                        critic_grad_norm = th.nn.utils.clip_grad_norm_(
                            self.policy.critic_parameters(),
                            getattr(
                                self.args, "matwm_agent_grad_clip", 100.0
                            ),
                        )
                        self.critic_optimiser.step()
                        grad_norm = th.maximum(
                            th.as_tensor(actor_grad_norm),
                            th.as_tensor(critic_grad_norm),
                        )
                    else:
                        self.agent_optimiser.zero_grad()
                        (
                            actor_loss + value_coefficient * critic_loss
                        ).backward()
                        grad_norm = th.nn.utils.clip_grad_norm_(
                            list(self.policy.actor_parameters())
                            + list(self.policy.critic_parameters()),
                            getattr(
                                self.args, "matwm_agent_grad_clip", 100.0
                            ),
                        )
                        self.agent_optimiser.step()
                    actor_losses.append(actor_loss.item())
                    critic_losses.append(critic_loss.item())
                    grad_norms.append(float(grad_norm))
            self.policy.update_ema(getattr(self.args, "matwm_ema_decay", 0.98))
            return {
                "matwm_actor_loss": sum(actor_losses) / len(actor_losses),
                "matwm_critic_loss": sum(critic_losses) / len(critic_losses),
                "matwm_real_critic_loss": 0.0,
                "matwm_return_mean": returns.mean().item(),
                "matwm_critic_return_mean": returns.mean().item(),
                "matwm_critic_return_std": returns.std(unbiased=False).item(),
                "matwm_imagination_continue": th.stack(
                    continuations, 1
                ).mean().item(),
                "matwm_agent_grad_norm": sum(grad_norms) / len(grad_norms),
            }
        actor_loss = -self._masked_mean(
            advantage * log_probs + entropy_coefficient * entropies, weights
        )
        ema_critic_coefficient = getattr(
            self.args, "matwm_critic_ema_coef", 1.0
        )
        critic_error = (values - returns.detach()).pow(2)
        if ema_critic_coefficient:
            critic_error = critic_error + ema_critic_coefficient * (
                values - ema_values.detach()
            ).pow(2)
        critic_loss = self._masked_mean(critic_error, weights)
        real_critic_coefficient = getattr(
            self.args, "matwm_real_critic_coef", 0.0
        )
        real_critic_loss = (values[:, 0] - real_returns.detach()).pow(2).mean()
        critic_loss = critic_loss + real_critic_coefficient * real_critic_loss
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
            "matwm_real_critic_loss": real_critic_loss.item(),
            "matwm_return_mean": returns.mean().item(),
            "matwm_critic_return_mean": returns.mean().item(),
            "matwm_critic_return_std": returns.std(unbiased=False).item(),
            "matwm_imagination_continue": th.stack(continuations, 1).mean().item(),
            "matwm_agent_grad_norm": float(grad_norm),
        }

    def cuda(self):
        self.mac.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        th.save(self.world_optimiser.state_dict(), os.path.join(path, "world_opt.th"))
        th.save(self.agent_optimiser.state_dict(), os.path.join(path, "agent_opt.th"))
        if self.separate_ppo_optimisers:
            th.save(self.actor_optimiser.state_dict(), os.path.join(path, "actor_opt.th"))
            th.save(self.critic_optimiser.state_dict(), os.path.join(path, "critic_opt.th"))

    def load_models(self, path):
        self.mac.load_models(path)
        self.world_optimiser.load_state_dict(
            th.load(os.path.join(path, "world_opt.th"), map_location="cpu")
        )
        self.agent_optimiser.load_state_dict(
            th.load(os.path.join(path, "agent_opt.th"), map_location="cpu")
        )
        if self.separate_ppo_optimisers:
            actor_path = os.path.join(path, "actor_opt.th")
            critic_path = os.path.join(path, "critic_opt.th")
            if os.path.exists(actor_path):
                self.actor_optimiser.load_state_dict(th.load(actor_path, map_location="cpu"))
            if os.path.exists(critic_path):
                self.critic_optimiser.load_state_dict(th.load(critic_path, map_location="cpu"))
