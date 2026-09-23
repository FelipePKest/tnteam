"""MARIE's staged tokenizer, world-model, and imagined policy training."""

import os
import time

import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, AdamW

from learners.matwm_learner import MATWMLearner
from modules.marie_convergence import ModelConvergence
from components.episode_buffer import EpisodeBatch


class MARIELearner(MATWMLearner):
    """EPyMARL adapter for the training schedule used by official MARIE."""

    def __init__(self, mac, scheme, logger, args):
        # Opt-in TF32 for CUDA matrix multiplication; keep existing precision by default.
        th.backends.cuda.matmul.allow_tf32 = bool(getattr(args, "marie_allow_tf32", False))
        super().__init__(mac, scheme, logger, args)
        tokenizer_parameters = (
            list(self.world_model.encoder.parameters())
            + list(self.world_model.decoder.parameters())
        )
        self.tokenizer_optimiser = AdamW(
            tokenizer_parameters,
            lr=getattr(args, "marie_tokenizer_lr", 3e-4),
            weight_decay=getattr(args, "marie_tokenizer_weight_decay", 0.01),
        )
        tokenizer_ids = {
            id(parameter) for parameter in tokenizer_parameters
        }
        world_parameters = [
            parameter for parameter in self.world_model.parameters()
            if id(parameter) not in tokenizer_ids
        ]
        # Upstream treats encoded token IDs as fixed targets while updating the
        # world model. Do not let the world optimizer move the tokenizer.
        world_ids = {id(parameter) for parameter in world_parameters}
        decay_ids = set()
        for module in self.world_model.modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.MultiheadAttention)):
                weight = getattr(module, "weight", None)
                if weight is not None and id(weight) in world_ids:
                    decay_ids.add(id(weight))
                # MultiheadAttention stores its combined projection directly.
                projection = getattr(module, "in_proj_weight", None)
                if projection is not None and id(projection) in world_ids:
                    decay_ids.add(id(projection))
        decay_parameters = [
            parameter for parameter in world_parameters
            if id(parameter) in decay_ids
        ]
        no_decay_parameters = [
            parameter for parameter in world_parameters
            if id(parameter) not in decay_ids
        ]
        world_weight_decay = getattr(args, "marie_world_weight_decay", 0.01)
        self.world_optimiser = AdamW(
            [
                {"params": decay_parameters, "weight_decay": world_weight_decay},
                {"params": no_decay_parameters, "weight_decay": 0.0},
            ],
            lr=getattr(args, "matwm_world_lr", 1e-4),
            eps=getattr(args, "marie_world_eps", 1e-8),
        )
        actor_parameters = list(self.policy.actor_parameters())
        critic_parameters = list(self.policy.critic_parameters())
        optimiser_kwargs = {
            "eps": getattr(args, "marie_actor_critic_eps", 1e-8),
            "weight_decay": getattr(
                args, "marie_actor_critic_weight_decay", 1e-5
            ),
        }
        self.agent_optimiser = Adam(
            actor_parameters + critic_parameters,
            lr=getattr(args, "matwm_agent_lr", 5e-4),
            **optimiser_kwargs,
        )
        if self.separate_ppo_optimisers:
            self.actor_optimiser = Adam(
                actor_parameters,
                lr=getattr(args, "matwm_agent_lr", 5e-4),
                **optimiser_kwargs,
            )
            self.critic_optimiser = Adam(
                critic_parameters,
                lr=getattr(args, "matwm_ppo_critic_lr", 5e-4),
                **optimiser_kwargs,
            )
        self.marie_update_events = 0
        self.adaptive_model_updates = getattr(args, "marie_adaptive_model_updates", False)
        self.model_update_interval = int(getattr(args, "marie_reduced_model_interval", 5))
        self.model_update_win_rate = float(getattr(args, "marie_model_win_rate_threshold", 0.25))
        self.model_update_patience = int(getattr(args, "marie_model_threshold_evaluations", 3))
        if self.model_update_interval < 1 or self.model_update_patience < 1:
            raise ValueError("MARIE update interval and evaluation patience must be positive")
        if not 0 <= self.model_update_win_rate <= 1:
            raise ValueError("MARIE win-rate threshold must be between zero and one")
        self.model_threshold_streak = 0
        self.model_reduced_since = None
        self.model_last_eval_t = None
        self.convergence = None
        self.convergence_batches = None
        if getattr(args, "marie_convergence_stop", False):
            if self.adaptive_model_updates:
                raise ValueError("Choose convergence stopping or win-rate scheduling, not both")
            self.convergence = ModelConvergence(
                window=getattr(args, "marie_convergence_window", 3),
                patience=getattr(args, "marie_convergence_patience", 2),
                parameter_tolerance=getattr(args, "marie_convergence_parameter_tolerance", 0.01),
                validation_tolerance=getattr(args, "marie_convergence_validation_tolerance", 0.02),
                min_events=getattr(args, "marie_convergence_min_events", 20),
            )
        self.policy_only = getattr(args, "marie_policy_only", False)
        if self.policy_only:
            self.world_model.requires_grad_(False)
            self.world_model.eval()

    def _train_tokenizer(self, batch):
        self.world_model.encoder.train()
        observation = batch["obs"]
        valid = batch["filled"].squeeze(-1).bool()
        observation = observation[valid]
        # Bound team timesteps, never individual agents within those teams.
        team_batch_size = getattr(self.args, "marie_tokenizer_batch_size", 256)
        if observation.shape[0] > team_batch_size:
            indices = th.randperm(observation.shape[0], device=observation.device)[:team_batch_size]
            observation = observation[indices]
        observation = observation.reshape(-1, batch["obs"].shape[-1])
        if observation.numel() == 0:
            self.world_model.encoder.eval()
            return {
                "marie_tokenizer_loss": 0.0,
                "marie_tokenizer_reconstruction_loss": 0.0,
                "marie_tokenizer_commitment_loss": 0.0,
                "marie_tokenizer_entropy": 0.0,
                "marie_tokenizer_active_codes": 0.0,
                "marie_tokenizer_grad_norm": 0.0,
            }
        # Replay already samples marie_tokenizer_batch_size team observations.
        # Keep every agent: 256 teams on 3s_vs_5z means 768 observations.
        latent, logits = self.world_model.encode(observation, sample=True)
        reconstruction = self.world_model.decode(latent)
        reconstruction_loss = (reconstruction - observation).abs().mean()
        commitment_loss = self.world_model.encoder.commitment_loss(observation)
        probabilities = logits.softmax(-1)
        entropy = -(probabilities * probabilities.clamp_min(1e-8).log()).sum(-1).mean()
        loss = reconstruction_loss + commitment_loss
        self.tokenizer_optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(
            list(self.world_model.encoder.parameters())
            + list(self.world_model.decoder.parameters()),
            getattr(self.args, "marie_tokenizer_grad_clip", 10.0),
        )
        self.tokenizer_optimiser.step()
        self.world_model.encoder.update_codebook(observation)
        self.world_model.encoder.eval()
        active = logits.argmax(-1).unique().numel()
        return {
            "marie_tokenizer_loss": loss.item(),
            "marie_tokenizer_reconstruction_loss": reconstruction_loss.item(),
            "marie_tokenizer_commitment_loss": commitment_loss.item(),
            "marie_tokenizer_entropy": entropy.item(),
            "marie_tokenizer_active_codes": float(active),
            "marie_tokenizer_grad_norm": float(grad_norm),
        }

    def _train_agents(self, batch):
        # no_grad alone does not disable dropout. Imagination must use the
        # inference-mode dynamics, as in upstream MARIE.
        modes = [(module, module.training) for module in self.world_model.modules()]
        self.world_model.eval()
        try:
            return super()._train_agents(batch)
        finally:
            for module, training in modes:
                module.training = training

    def _train_world_model(self, batch, validate_rollout=False):
        """Train the four prediction heads used by upstream MARIE.

        MATWM adds reconstruction, teammate modelling and balanced KL terms.
        Those are useful MATWM objectives, but they are not part of MARIE's
        world-model loss.  MARIE freezes its VQ tokenizer and learns next VQ
        codes, reward, continuation and the available-action mask.
        """
        # Enable regularization only for supervised world-model optimization.
        self.world_model.train()
        self.world_model.encoder.eval()
        tokenizer_parameters = (
            list(self.world_model.encoder.parameters())
            + list(self.world_model.decoder.parameters())
        )
        previous = [parameter.requires_grad for parameter in tokenizer_parameters]
        for parameter in tokenizer_parameters:
            parameter.requires_grad_(False)
        try:
            obs = batch["obs"]
            actions = batch["actions"]
            available = batch["avail_actions"].float()
            rewards = batch["reward"]
            terminated = batch["terminated"].float()
            valid = batch["filled"][:, :-1].float()
            batch_size, total_t, n_agents, obs_dim = obs.shape
            length = total_t - 1
            if length < 1:
                return {"matwm_world_loss": 0.0}

            max_length = self.world_model.max_seq_length
            if length > max_length:
                start = int(th.randint(
                    length - max_length + 1, (1,), device=obs.device
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

            focal = th.arange(n_agents, device=obs.device).repeat(batch_size)
            focal_obs = obs.permute(0, 2, 1, 3).reshape(
                batch_size * n_agents, total_t, obs_dim
            )
            focal_actions = actions.permute(0, 2, 1, 3).reshape(
                batch_size * n_agents, total_t, 1
            )
            focal_available = available.permute(0, 2, 1, 3).reshape(
                batch_size * n_agents, total_t, self.n_actions
            )
            focal_valid = valid[:, None].expand(
                -1, n_agents, -1, -1
            ).reshape(batch_size * n_agents, length, 1)

            with th.no_grad():
                latent, _ = self.world_model.encode(focal_obs, sample=False)
            reference_layout = (
                getattr(self.args, "marie_original_procedure", False)
                and hasattr(
                    self.world_model, "reference_teacher_forced_dynamics"
                )
            )
            if hasattr(self.world_model, "teacher_forced_dynamics"):
                # Upstream DreamerMemory masks attention across every done
                # boundary, separately for each agent stream.
                team_done = terminated[:, :length].bool().squeeze(-1)
                token_count = length * self.world_model.block_size
                if not reference_layout:
                    token_count += self.world_model.n_latents
                attention_mask = th.zeros(
                    batch_size, token_count, token_count,
                    dtype=th.bool, device=obs.device,
                )
                for episode in range(batch_size):
                    begin = 0
                    boundaries = (
                        team_done[episode, :-1].nonzero().flatten() + 1
                    ).tolist()
                    ends = boundaries + [length]
                    for boundary_index, end in enumerate(ends):
                        left = begin * self.world_model.block_size
                        right = (
                            token_count
                            if boundary_index == len(ends) - 1
                            else end * self.world_model.block_size
                        )
                        attention_mask[episode, left:right, left:right] = th.tril(
                            th.ones(
                                right - left, right - left,
                                dtype=th.bool, device=obs.device,
                            )
                        )
                        begin = end
                attention_mask = attention_mask.repeat_interleave(
                    n_agents, dim=0
                )
                if reference_layout:
                    hidden, dynamics_logits = (
                        self.world_model.reference_teacher_forced_dynamics(
                            latent[:, :length], focal_actions[:, :length],
                            focal, attention_mask,
                        )
                    )
                else:
                    hidden, dynamics_logits = (
                        self.world_model.teacher_forced_dynamics(
                            latent, focal_actions[:, :-1], focal,
                            attention_mask,
                        )
                    )
                heads = self.world_model.prediction_auxiliary_heads(
                    hidden, latent[:, :-1]
                )
                heads["dynamics_logits"] = dynamics_logits
            else:
                hidden = self.world_model.dynamics_sequence(
                    latent[:, :-1], focal_actions[:, :-1], focal
                )
                heads = self.world_model.prediction_heads(
                    hidden, latent[:, :-1], latent[:, 1:]
                )

            if reference_layout:
                token_target = latent[:, :length].argmax(-1).reshape(
                    batch_size * n_agents, -1
                )[:, 1:]
                token_loss_value = F.cross_entropy(
                    heads["dynamics_logits"].reshape(
                        -1, self.world_model.n_categories
                    ),
                    token_target.reshape(-1),
                )
                token_loss = token_loss_value.expand(
                    batch_size * n_agents, length, 1
                )
            else:
                token_target = latent[:, 1:].argmax(-1)
                token_loss = F.cross_entropy(
                    heads["dynamics_logits"].reshape(
                        -1, self.world_model.n_categories
                    ),
                    token_target.reshape(-1), reduction="none",
                ).view(
                    batch_size * n_agents, length, -1
                ).mean(-1, keepdim=True)

            reward_target = rewards[:, :length, None].expand(
                -1, -1, n_agents, -1
            ).permute(0, 2, 1, 3).reshape(batch_size * n_agents, length, 1)
            reward_logits = heads["reward_ensemble_logits"][0]
            if self.world_model.reward_regression:
                reward_loss = F.smooth_l1_loss(
                    reward_logits, reward_target, reduction="none"
                )
            else:
                reward_distribution = self.world_model.two_hot_reward(reward_target)
                reward_loss = -(
                    reward_distribution * reward_logits.log_softmax(-1)
                ).sum(-1, keepdim=True)

            continue_target = (1.0 - terminated[:, :length])[:, None].expand(
                -1, n_agents, -1, -1
            ).reshape(batch_size * n_agents, length, 1)
            continuation_loss = F.binary_cross_entropy_with_logits(
                heads["continuation_logits"], continue_target, reduction="none"
            )
            # Canonical MARIE applies the Bernoulli objective uniformly. Keep
            # an explicit weight only as a backwards-compatible override.
            terminal_weight = getattr(
                self.args, "marie_terminal_loss_weight", 1.0
            )
            continuation_loss = continuation_loss * th.where(
                continue_target.bool(), 1.0, terminal_weight
            )
            if reference_layout:
                availability_steps = length - 1
                availability_loss = heads["availability_logits"].new_zeros(
                    batch_size * n_agents, length, 1
                )
                if availability_steps:
                    availability_raw = F.cross_entropy(
                        heads["availability_logits"][
                            :, :availability_steps
                        ].reshape(-1, 2),
                        focal_available[:, 1:length].long().reshape(-1),
                        reduction="none",
                    ).view(
                        batch_size * n_agents, availability_steps,
                        self.n_actions,
                    ).mean(-1, keepdim=True)
                    availability_loss[:, :availability_steps] = (
                        availability_raw * length / availability_steps
                    )
            else:
                availability_loss = F.cross_entropy(
                    heads["availability_logits"].reshape(-1, 2),
                    focal_available[:, 1:].long().reshape(-1),
                    reduction="none",
                ).view(
                    batch_size * n_agents, length, self.n_actions
                ).mean(-1, keepdim=True)

            total = token_loss + reward_loss + continuation_loss + availability_loss
            loss = self._masked_mean(total, focal_valid)
            self.world_optimiser.zero_grad()
            loss.backward()
            grad_norm = th.nn.utils.clip_grad_norm_(
                [parameter for group in self.world_optimiser.param_groups
                 for parameter in group["params"]],
                getattr(self.args, "matwm_world_grad_clip", 100.0),
            )
            self.world_optimiser.step()
            with th.no_grad():
                predicted_continue = heads["continuation_logits"].sigmoid() >= 0.5
                terminal_target = ~continue_target.bool()
                predicted_terminal = ~predicted_continue
                terminal_count = (terminal_target * focal_valid.bool()).sum()
                terminal_recall = (
                    (predicted_terminal & terminal_target & focal_valid.bool()).sum()
                    / terminal_count.clamp_min(1)
                )
                availability_logits = heads["availability_logits"]
                availability_target = focal_available[:, 1:].bool()
                if reference_layout:
                    availability_logits = availability_logits[:, :-1]
                    availability_target = focal_available[:, 1:length].bool()
                availability_prediction = self.world_model.availability_from_logits(
                    availability_logits
                )
                availability_correct = (
                    availability_prediction == availability_target
                ).float().mean(-1, keepdim=True)
                if reference_layout:
                    token_accuracy_value = (
                        heads["dynamics_logits"].argmax(-1) == token_target
                    ).float().mean()
                    token_accuracy = token_accuracy_value.expand(
                        batch_size * n_agents, length, 1
                    )
                else:
                    token_accuracy = (
                        heads["dynamics_logits"].argmax(-1) == token_target
                    ).float().mean(-1, keepdim=True)
                # Open-loop validation: feed generated tokens back into the
                # model while retaining the recorded actions. This exposes
                # compounding errors hidden by teacher-forced CE.
                rollout_correct = []
                if validate_rollout:
                    rollout_target = latent[:, 1:].argmax(-1)
                    rollout_horizon = min(
                        getattr(self.args, "marie_validation_horizon", 5), length
                    )
                    rollout_latent = latent[:, :1]
                    rollout_actions = focal_actions[:, :0]
                    for rollout_step in range(rollout_horizon):
                        rollout_actions = th.cat((
                            rollout_actions,
                            focal_actions[:, rollout_step:rollout_step + 1],
                        ), dim=1)
                        rollout_hidden = self.world_model.dynamics_sequence(
                            rollout_latent, rollout_actions, focal
                        )[:, -1]
                        rollout_logits = (
                            self.world_model.autoregressive_dynamics_logits(
                                rollout_hidden
                            )
                        )
                        rollout_index = rollout_logits.argmax(-1)
                        rollout_correct.append((
                            rollout_index == rollout_target[:, rollout_step]
                        ).float().mean())
                        rollout_next = F.one_hot(
                            rollout_index, self.world_model.n_categories
                        ).to(latent.dtype)
                        rollout_latent = th.cat(
                            (rollout_latent, rollout_next[:, None]), dim=1
                        )
            stats = {
                "matwm_world_loss": loss.item(),
                "matwm_dynamics_loss": self._masked_mean(
                    token_loss, focal_valid
                ).item(),
                "matwm_reward_loss": self._masked_mean(
                    reward_loss, focal_valid
                ).item(),
                "matwm_continuation_loss": self._masked_mean(
                    continuation_loss, focal_valid
                ).item(),
                "matwm_mask_loss": self._masked_mean(
                    availability_loss, focal_valid
                ).item(),
                "marie_token_accuracy_1step": self._masked_mean(
                    token_accuracy, focal_valid
                ).item(),
                "marie_availability_accuracy": self._masked_mean(
                    availability_correct,
                    focal_valid[:, :availability_correct.shape[1]],
                ).item(),
                "marie_terminal_recall": terminal_recall.item(),
                "matwm_world_grad_norm": float(grad_norm),
            }
            if rollout_correct:
                stats["marie_token_accuracy_open_loop"] = th.stack(
                    rollout_correct
                ).mean().item()
                stats["marie_token_accuracy_open_loop_last"] = (
                    rollout_correct[-1].item()
                )
            return stats
        finally:
            self.world_model.eval()
            for parameter, requires_grad in zip(tokenizer_parameters, previous):
                parameter.requires_grad_(requires_grad)

    @staticmethod
    def _average_stats(stats):
        if not stats:
            return {}
        return {
            key: sum(item[key] for item in stats if key in item)
                 / sum(key in item for item in stats)
            for key in set().union(*(item.keys() for item in stats))
        }

    def _replay_batch(self, replay, batch_size, sequence_length, mode):
        batch = replay.sample_marie(
            batch_size,
            sequence_length,
            mode=mode,
            temperature=getattr(self.args, "marie_sample_temperature", "inf"),
        )
        if batch.device != th.device(self.args.device):
            batch.to(self.args.device)
        return batch

    def observe_evaluation(self, win_rate, t_env):
        """Latch a slower model schedule after consecutive qualifying evaluations."""
        if not self.adaptive_model_updates or self.policy_only:
            return
        if self.model_last_eval_t == t_env:
            return
        self.model_last_eval_t = t_env
        if self.model_reduced_since is None:
            self.model_threshold_streak = (
                self.model_threshold_streak + 1
                if win_rate >= self.model_update_win_rate else 0
            )
            if self.model_threshold_streak >= self.model_update_patience:
                self.model_reduced_since = self.marie_update_events
                console = getattr(self.logger, "console_logger", None)
                if console is not None:
                    console.info(
                        "MARIE model schedule reduced at t_env=%s: win rate %.3f "
                        "met %.3f for %s evaluations; tokenizer/WM every %s events",
                        t_env, win_rate, self.model_update_win_rate,
                        self.model_update_patience, self.model_update_interval,
                    )
        self.logger.log_stat("marie_model_threshold_streak", self.model_threshold_streak, t_env)
        self.logger.log_stat("marie_model_update_interval", self._model_interval(), t_env)

    def _model_interval(self):
        if self.adaptive_model_updates and self.model_reduced_since is not None:
            return self.model_update_interval
        return 1

    def _update_model_this_event(self):
        if self.policy_only:
            return False
        if self._model_interval() == 1:
            return True
        return (self.marie_update_events - self.model_reduced_since) % self.model_update_interval == 0

    def observe_model_convergence(self, batches, t_env):
        if self.convergence is None or self.policy_only:
            return
        if self.convergence_batches is None:
            # Keep the first held-out episodes fixed for comparable measurements.
            # Store plain tensor dictionaries so they can be checkpointed portably.
            self.convergence_batches = []
            for batch in batches:
                self.convergence_batches.append({
                    "scheme": {k:v for k,v in batch.scheme.items() if k != "filled"},
                    "groups": batch.groups, "batch_size": batch.batch_size,
                    "max_seq_length": batch.max_seq_length,
                    "transitions": {k:v.detach().cpu().clone() for k,v in batch.data.transition_data.items()},
                    "episodes": {k:v.detach().cpu().clone() for k,v in batch.data.episode_data.items()},
                })
        fixed = []
        for saved in self.convergence_batches:
            batch = EpisodeBatch(saved["scheme"], saved["groups"], saved["batch_size"], saved["max_seq_length"])
            batch.data.transition_data = saved["transitions"]
            batch.data.episode_data = saved["episodes"]
            fixed.append(batch)
        stats = self.validate_world_model(fixed, t_env, prefix="marie_fixed_validation")
        keys = ["tokenizer_reconstruction_mae"] + [
            "h%d/%s" % (h, metric) for h in (1, 5, 15)
            for metric in ("observation_mae", "reward_mae", "return_mae")
        ]
        if any("marie_fixed_validation/"+key not in stats for key in keys):
            return  # insufficient held-out horizon; never infer convergence
        metrics = {key: stats["marie_fixed_validation/"+key] for key in keys}
        diagnostics, frozen = self.convergence.observe(
            ModelConvergence.snapshot(self.world_model), metrics, self.marie_update_events)
        for key, value in diagnostics.items():
            self.logger.log_stat("marie_convergence/"+key, value, t_env)
        if frozen:
            self.policy_only = True
            self.world_model.requires_grad_(False)
            self.world_model.eval()
            self.logger.console_logger.info(
                "MARIE convergence plateau at t_env=%s: freezing tokenizer and world model; "
                "continuing agent-only training. This is a heuristic plateau, not proof of optimality.", t_env)

    @th.no_grad()
    def validate_world_model(self, batches, t_env, prefix="marie_validation"):
        """Open-loop predictions on fresh evaluation episodes, never replayed.

        Follow recorded actions with the same cached stochastic dynamics used
        by policy imagination. Report each horizon separately; no universal
        sufficiency threshold is assumed. Preserve training RNG and modes.
        """
        modes = [(module, module.training) for module in self.world_model.modules()]
        records = {h: [] for h in (1, 5, 15)}
        reconstruction_errors = []
        device = next(self.world_model.parameters()).device
        devices = [device.index or 0] if device.type == "cuda" else []
        try:
            self.world_model.eval()
            with th.random.fork_rng(devices=devices):
                th.manual_seed(1729)
                for batch in batches:
                    for episode in range(batch.batch_size):
                        length = int(batch["filled"][episode].sum().item()) - 1
                        if length < 1:
                            continue
                        # Cover beginnings, middles and terminal boundaries.
                        width = min(15, length)
                        starts = sorted(set((0, (length-width)//2, length-width)))
                        for start in starts:
                            obs = batch["obs"][episode, start:start+width+1].to(device)
                            actions = batch["actions"][episode, start:start+width].to(device)
                            rewards = batch["reward"][episode, start:start+width].to(device)
                            done = batch["terminated"][episode, start:start+width].to(device)
                            avail = batch["avail_actions"][episode, start:start+width+1].to(device)
                            target, _ = self.world_model.encode(obs.transpose(0, 1), sample=False)
                            reconstruction_errors.append((
                                self.world_model.decode(target) - obs.transpose(0, 1)
                            ).abs().mean().item())
                            latent = target[:, :1]
                            focal = th.arange(self.n_agents, device=device)
                            cache = None
                            predicted_return = th.zeros(self.n_agents, 1, device=device)
                            real_return = th.zeros_like(predicted_return)
                            for step in range(width):
                                action = actions[step][:, None]
                                if cache is None:
                                    hidden, cache = self.world_model.init_dynamics_cache(latent, action, focal)
                                    hidden = hidden[:, -1]
                                else:
                                    hidden, cache = self.world_model.append_action_cache(latent, action, focal, cache)
                                heads = self.world_model.prediction_heads(hidden, latent[:, -1])
                                reward = self.world_model.reward_value(heads["reward_ensemble_logits"][0])
                                predicted_return += reward
                                real_return += rewards[step]
                                terminal = heads["continuation_logits"].sigmoid() < 0.5
                                terminal_target = done[step].bool().expand_as(terminal)
                                availability = self.world_model.predicted_availability_from_hidden(hidden)
                                next_latent, _, cache = self.world_model.sample_next_latent_cached(hidden, cache)
                                horizon = step + 1
                                if horizon in records:
                                    records[horizon].append({
                                        "observation_mae": (self.world_model.decode(next_latent)-obs[horizon]).abs().mean().item(),
                                        "token_accuracy": (next_latent.argmax(-1)==target[:, horizon].argmax(-1)).float().mean().item(),
                                        "reward_mae": (reward-rewards[step]).abs().mean().item(),
                                        "return_mae": (predicted_return-real_return).abs().mean().item(),
                                        "availability_accuracy": (availability.bool()==avail[horizon].bool()).float().mean().item(),
                                        "terminal_tp": (terminal & terminal_target).sum().item(),
                                        "terminal_predicted": terminal.sum().item(),
                                        "terminal_actual": terminal_target.sum().item(),
                                    })
                                latent = next_latent[:, None]
        finally:
            for module, training in modes:
                module.training = training
        result = {}
        base_prefix = prefix
        if reconstruction_errors:
            result[base_prefix+"/tokenizer_reconstruction_mae"] = sum(reconstruction_errors)/len(reconstruction_errors)
        for horizon, rows in records.items():
            if not rows:
                continue
            prefix = base_prefix + "/h%d/" % horizon
            for key in rows[0]:
                if not key.startswith("terminal_"):
                    result[prefix+key] = sum(row[key] for row in rows)/len(rows)
            tp = sum(row["terminal_tp"] for row in rows)
            predicted = sum(row["terminal_predicted"] for row in rows)
            actual = sum(row["terminal_actual"] for row in rows)
            result[prefix+"terminal_actual"] = actual
            result[prefix+"terminal_predicted"] = predicted
            if predicted:
                result[prefix+"terminal_precision"] = tp/predicted
            if actual:
                result[prefix+"terminal_recall"] = tp/actual
            result[prefix+"windows"] = len(rows)
        for key, value in result.items():
            self.logger.log_stat(key, value, t_env)
        return result

    def _stage_time(self):
        device = th.device(self.args.device)
        if device.type == "cuda":
            th.cuda.synchronize(device)
        return time.perf_counter()

    def train_from_replay(self, replay, t_env, episode_num):
        """Run one canonical MARIE update event using fresh replay draws."""
        self.train_calls += 1
        self.marie_update_events += 1
        update_model = self._update_model_this_event()
        self.logger.log_stat("marie_model_updated", int(update_model), t_env)
        self.logger.log_stat("marie_model_update_interval", self._model_interval(), t_env)
        started = self._stage_time()
        tokenizer_stats = []
        tokenizer_batch_size = getattr(
            self.args, "marie_tokenizer_batch_size", 256
        )
        for _ in range(0 if not update_model else getattr(self.args, "marie_tokenizer_epochs", 200)):
            batch = self._replay_batch(
                replay, tokenizer_batch_size, 0, "tokenizer"
            )
            tokenizer_stats.append(self._train_tokenizer(batch))

        tokenizer_finished = self._stage_time()
        world_stats = []
        if update_model and self.marie_update_events > getattr(
            self.args, "marie_world_warmup_events", 9
        ):
            world_epochs = getattr(self.args, "marie_world_epochs", 200)
            for world_epoch in range(world_epochs):
                batch = self._replay_batch(
                    replay,
                    getattr(self.args, "batch_size", 30),
                    getattr(self.args, "matwm_max_seq_length", 15),
                    "model",
                )
                world_stats.append(self._train_world_model(
                    batch, validate_rollout=(world_epoch == world_epochs - 1)
                ))
        averaged_world = self._average_stats(world_stats)
        if averaged_world:
            self.last_world_stats = averaged_world

        world_finished = self._stage_time()
        agent_stats = []
        if self.marie_update_events > getattr(
            self.args, "marie_policy_warmup_events", 19
        ):
            for _ in range(getattr(self.args, "marie_policy_epochs", 5)):
                # Five observations provide the upstream four-history-plus-
                # current policy stack. Draw contexts from the full replay.
                batch = self._replay_batch(
                    replay,
                    getattr(self.args, "matwm_agent_batch_size", 600),
                    getattr(self.args, "marie_stack_obs", 5) - 1,
                    "policy",
                )
                agent_stats.append(self._train_agents(batch))
        averaged_agent = self._average_stats(agent_stats)
        if averaged_agent:
            self.last_agent_stats = averaged_agent

        policy_finished = self._stage_time()
        # Synchronize only at stage boundaries; include replay draws and
        # transfers in each stage's wall time instead of timing GPU launches.
        for key, seconds in {
            "tokenizer": tokenizer_finished - started,
            "world_model": world_finished - tokenizer_finished,
            "imagined_policy": policy_finished - world_finished,
            "total": policy_finished - started,
        }.items():
            self.logger.log_stat("marie_seconds_" + key, seconds, t_env)

        stats = {
            **self._average_stats(tokenizer_stats),
            **(averaged_world or self.last_world_stats or {}),
            **(averaged_agent or self.last_agent_stats or {}),
        }
        if t_env - self.last_log_t >= self.args.learner_log_interval:
            for key, value in stats.items():
                self.logger.log_stat(key, value, t_env)
            self.last_log_t = t_env

    def train(self, batch, t_env, episode_num):
        self.train_calls += 1
        self.marie_update_events += 1
        update_model = self._update_model_this_event()
        self.logger.log_stat("marie_model_updated", int(update_model), t_env)
        self.logger.log_stat("marie_model_update_interval", self._model_interval(), t_env)
        tokenizer_stats = [
            self._train_tokenizer(batch)
            for _ in range(0 if not update_model else getattr(self.args, "marie_tokenizer_epochs", 1))
        ]

        world_stats = []
        if update_model and self.marie_update_events > getattr(
            self.args, "marie_world_warmup_events", 0
        ):
            world_stats = [
                self._train_world_model(batch)
                for _ in range(getattr(self.args, "marie_world_epochs", 1))
            ]
        averaged_world = self._average_stats(world_stats)
        if averaged_world:
            self.last_world_stats = averaged_world

        agent_stats = []
        if (
            t_env >= getattr(self.args, "matwm_prefill_steps", 1000)
            and self.marie_update_events > getattr(
                self.args, "marie_policy_warmup_events", 0
            )
        ):
            agent_stats = [
                self._train_agents(batch)
                for _ in range(getattr(self.args, "marie_policy_epochs", 1))
            ]
        averaged_agent = self._average_stats(agent_stats)
        if averaged_agent:
            self.last_agent_stats = averaged_agent

        stats = {
            **self._average_stats(tokenizer_stats),
            **(averaged_world or self.last_world_stats or {}),
            **(averaged_agent or self.last_agent_stats or {}),
        }
        if t_env - self.last_log_t >= self.args.learner_log_interval:
            for key, value in stats.items():
                self.logger.log_stat(key, value, t_env)
            self.last_log_t = t_env

    def save_models(self, path):
        super().save_models(path)
        th.save(
            self.tokenizer_optimiser.state_dict(),
            os.path.join(path, "tokenizer_opt.th"),
        )
        th.save(
            {
                "marie_update_events": self.marie_update_events,
                "train_calls": self.train_calls,
                "model_threshold_streak": self.model_threshold_streak,
                "model_reduced_since": self.model_reduced_since,
                "model_last_eval_t": self.model_last_eval_t,
                "convergence": self.convergence.state_dict() if self.convergence else None,
                "convergence_batches": self.convergence_batches,
            },
            os.path.join(path, "marie_training_state.th"),
        )

    def load_models(self, path):
        if (
            not os.path.exists(os.path.join(path, "agent.th"))
            and os.path.exists(os.path.join(path, "reference_marie.pt"))
        ):
            self.mac.load_models(path)
            return
        super().load_models(path)
        tokenizer_path = os.path.join(path, "tokenizer_opt.th")
        if os.path.exists(tokenizer_path):
            self.tokenizer_optimiser.load_state_dict(
                th.load(tokenizer_path, map_location="cpu")
            )
        state_path = os.path.join(path, "marie_training_state.th")
        if os.path.exists(state_path):
            state = th.load(state_path, map_location="cpu")
            self.marie_update_events = state.get("marie_update_events", 0)
            self.train_calls = state.get("train_calls", 0)
            self.model_threshold_streak = state.get("model_threshold_streak", 0)
            self.model_reduced_since = state.get("model_reduced_since")
            self.model_last_eval_t = state.get("model_last_eval_t")

            if self.convergence is not None and state.get("convergence") is not None:
                self.convergence.load_state_dict(state["convergence"])
                self.convergence_batches = state.get("convergence_batches")
                if self.convergence.frozen:
                    self.policy_only = True
                    self.world_model.requires_grad_(False)
                    self.world_model.eval()
