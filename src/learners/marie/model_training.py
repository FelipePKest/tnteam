"""MARIE model training implementation."""

import torch as th
import torch.nn.functional as F
from components.marie_context import controlled_training
from modules.world_models.marie.short_context import short_context_loss


class MARIEModelTraining:
    """MARIE model training methods shared by the learner adapter."""

    @controlled_training
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

    @controlled_training
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
            training_mask = batch.data.transition_data.get('trainable_agents')
            episode_end = batch.data.transition_data.get('episode_end', terminated)
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
                if training_mask is not None:
                    training_mask = training_mask[:,start:stop]
                episode_end = episode_end[:, start:stop]
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
                team_done = episode_end[:, :length].bool().squeeze(-1)
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
                recurrent_options = {}
                if getattr(self.world_model, 'recurrent_world_model', False):
                    recurrent_options = dict(
                        episode_end=team_done.repeat_interleave(n_agents, dim=0),
                        valid=focal_valid.squeeze(-1).bool(),
                    )
                if reference_layout:
                    hidden, dynamics_logits = (
                        self.world_model.reference_teacher_forced_dynamics(
                            latent[:, :length], focal_actions[:, :length],
                            focal, attention_mask, **recurrent_options,
                        )
                    )
                else:
                    hidden, dynamics_logits = (
                        self.world_model.teacher_forced_dynamics(
                            latent, focal_actions[:, :-1], focal,
                            attention_mask, **recurrent_options,
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
                    reduction="none",
                )
                token_valid = th.ones_like(token_target, dtype=th.bool)
                if 'episode_end' in batch.scheme:
                    token_valid = focal_valid.bool().expand(
                        -1, -1, self.world_model.n_latents
                    ).clone()
                    # The first code after reset has no physical predecessor.
                    # Later codes within that observation remain valid targets.
                    boundaries = episode_end[:, :length-1, 0].bool().repeat_interleave(
                        n_agents, dim=0
                    )
                    token_valid[:, 1:, 0] &= ~boundaries
                    token_valid = token_valid.flatten(1)[:, 1:]
                token_loss_value = self._masked_mean(
                    token_loss_value.view_as(token_target), token_valid
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
            foreign_stats = {}
            if getattr(self.args,'marie_full_team_world_model',False):
                controlled_mask = training_mask[:,:length].permute(0,2,1,3).reshape(
                    batch_size*n_agents,length,1).float()
                foreign_valid = focal_valid * (1-controlled_mask)
                foreign_logits = self.world_model.foreign_action_logits(latent[:,:length])
                foreign_loss = F.cross_entropy(foreign_logits.reshape(-1,self.n_actions),
                    focal_actions[:,:length,0].reshape(-1),reduction='none').reshape(-1,length,1)
                foreign_loss = self._masked_mean(foreign_loss,foreign_valid)
                loss = loss + foreign_loss
                foreign_stats = {'marie_foreign_action_loss':foreign_loss.item(),
                    'marie_foreign_action_accuracy':self._masked_mean((foreign_logits.argmax(-1)==
                        focal_actions[:,:length,0]).float()[...,None],foreign_valid).item(),
                    'marie_world_model_agents':float(n_agents)}
            short_weight = float(getattr(self.args, 'marie_short_context_loss_weight', 0.0))
            if short_weight < 0:
                raise ValueError('marie_short_context_loss_weight must be nonnegative')
            short_stats = None
            if short_weight:
                # Same batch and optimizer step; optional normalized mixture
                # of canonical supervision and rollout-initialized supervision.
                # The helper reads the original valid batch (not cropped latent
                # tensors), keeping reset/padding indices aligned.
                short_stats = short_context_loss(self.world_model, batch)
                if short_stats is not None:
                    loss = (loss + short_weight * short_stats['loss']) / (1 + short_weight)
            self.world_optimiser.zero_grad()
            loss.backward()
            alignment_stats = {}
            if getattr(self, '_alignment_active', False):
                from modules.world_models.marie.value_alignment import alignment_loss
                devices = [obs.device.index or 0] if obs.is_cuda else []
                with th.random.fork_rng(devices=devices):
                    th.manual_seed(self._alignment_seed)
                    auxiliary = alignment_loss(self.policy, self._alignment_guide, batch,
                        draws=int(getattr(self.args, 'marie_value_alignment_draws', 4)),
                        max_teams=int(getattr(self.args, 'marie_value_alignment_teams', 4)))
                if auxiliary is not None:
                    parameters = [p for group in self.world_optimiser.param_groups
                                  for p in group['params'] if p.requires_grad]
                    extra = th.autograd.grad(auxiliary['surrogate'], parameters, allow_unused=True)
                    base_norm = th.sqrt(sum(p.grad.detach().square().sum()
                                    for p in parameters if p.grad is not None))
                    extra_norm = th.sqrt(sum(g.detach().square().sum()
                                     for g in extra if g is not None))
                    weight = float(getattr(self.args, 'marie_value_alignment_weight', 0.0))
                    ramp = min(1., max(0., (self.marie_update_events-9)/20.))
                    coefficient = min(weight*ramp, float(getattr(
                        self.args, 'marie_value_alignment_grad_ratio', .1))
                        *float(base_norm)/(float(extra_norm)+1e-8))
                    for parameter, gradient in zip(parameters, extra):
                        if gradient is not None:
                            if parameter.grad is None:parameter.grad = coefficient*gradient.detach()
                            else:parameter.grad.add_(gradient.detach(), alpha=coefficient)
                    alignment_stats = {'marie_value_alignment/'+key:float(value)
                        for key,value in auxiliary.items() if key!='surrogate'}
                    alignment_stats.update({'marie_value_alignment/coefficient':coefficient,
                        'marie_value_alignment/gradient_ratio':coefficient*float(extra_norm)/(float(base_norm)+1e-8)})
            recurrent_grad_norm = None
            if getattr(self.world_model, 'recurrent_world_model', False):
                recurrent_grad_norm = th.linalg.vector_norm(th.stack([
                    p.grad.detach().norm() for p in self.world_model.recurrent_memory.parameters()
                    if p.grad is not None
                ]))
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
                    token_accuracy_value = self._masked_mean((
                        heads["dynamics_logits"].argmax(-1) == token_target
                    ).float(), token_valid)
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
                        getattr(self.args, "marie_validation_horizon", 5),
                        length - int(getattr(batch, 'marie_reference_records', False)),
                    )
                    rollout_alive = th.ones(batch_size * n_agents, dtype=th.bool, device=obs.device)
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
                        # A reset observation after done is not a physical
                        # next-observation target for an open-loop rollout.
                        rollout_alive &= ~episode_end[:, rollout_step, 0].bool().repeat_interleave(n_agents)
                        if rollout_alive.any():
                            rollout_correct.append((
                                rollout_index[rollout_alive] == rollout_target[rollout_alive, rollout_step]
                            ).float().mean())
                        rollout_next = F.one_hot(
                            rollout_index, self.world_model.n_categories
                        ).to(latent.dtype)
                        rollout_latent = th.cat(
                            (rollout_latent, rollout_next[:, None]), dim=1
                        )
            stats = {
                **foreign_stats,
                **alignment_stats,
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
            if recurrent_grad_norm is not None:
                stats['marie_recurrent_grad_norm'] = recurrent_grad_norm.item()
            if short_stats is not None:
                stats.update({
                    'marie_short_context_loss': short_stats['loss'].item(),
                    'marie_short_context_token_accuracy': short_stats['token_accuracy'].item(),
                    'marie_short_context_teams': float(short_stats['teams']),
                })
            return stats
        finally:
            self.world_model.eval()
            for parameter, requires_grad in zip(tokenizer_parameters, previous):
                parameter.requires_grad_(requires_grad)
