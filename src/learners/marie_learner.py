"""MARIE's staged tokenizer, world-model, and imagined policy training."""

import os
import time

import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, AdamW

from learners.matwm_learner import MATWMLearner
from modules.world_models.marie.convergence import ModelConvergence, ValidationPlateau
from modules.world_models.marie.validation import collect_validation
from components.marie_context import compact_controlled, controlled_group, controlled_training
from components.episode_buffer import EpisodeBatch


from learners.marie.model_training import MARIEModelTraining
from learners.marie.validation import MARIEValidation
from learners.marie.checkpoint import MARIECheckpoint


class MARIELearner(MARIEModelTraining, MARIEValidation, MARIECheckpoint, MATWMLearner):
    """Registered NAHT learner: optimizers, replay scheduling and policy updates."""

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
        self.set_optimizer_foreach(getattr(args, "marie_optimizer_foreach", False))
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
        self._controlled_context_counts = {}
        self.convergence_mode = getattr(args, "marie_convergence_mode", "parameters")
        if self.convergence_mode not in ("parameters", "validation"):
            raise ValueError("MARIE convergence mode must be parameters or validation")
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
            if self.convergence_mode == "validation":
                self.convergence = ValidationPlateau(
                    window=getattr(args, "marie_loss_window", 3),
                    patience=getattr(args, "marie_loss_patience", 6),
                    min_delta=getattr(args, "marie_loss_min_delta", .02),
                    min_events=getattr(args, "marie_convergence_min_events", 20),
                    deterioration=getattr(args, "marie_loss_deterioration", .15),
                    recovery_patience=getattr(args, "marie_loss_recovery_patience", 3),
                )
        self.policy_only = getattr(args, "marie_policy_only", False)
        if self.policy_only:
            self.world_model.requires_grad_(False)
            self.world_model.eval()

    @controlled_training
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
        options = {}
        if getattr(self.args, "marie_controlled_replay_only", False):
            options["controlled_only"] = True
        batch = replay.sample_marie(
            batch_size,
            sequence_length,
            mode=mode,
            temperature=getattr(self.args, "marie_sample_temperature", "inf"),
            **options,
        )
        if batch.device != th.device(self.args.device):
            batch.to(self.args.device)
        if getattr(self.args, "marie_controlled_replay_only", False):
            self._controlled_context_counts.setdefault(mode, []).append(batch["obs"].shape[2])
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

    def _stage_time(self):
        device = th.device(self.args.device)
        if device.type == "cuda":
            th.cuda.synchronize(device)
        return time.perf_counter()

    def train_from_replay(self, replay, t_env, episode_num):
        """Run one canonical MARIE update event using fresh replay draws."""
        self._controlled_context_counts = {}
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

        for mode, sizes in self._controlled_context_counts.items():
            self.logger.log_stat("marie_replay_" + mode + "_agents_per_context",
                                 sum(sizes) / len(sizes), t_env)

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

    def set_optimizer_foreach(self, enabled):
        """Apply runtime optimizer grouping, including to restored parameter groups."""
        for name in ("tokenizer_optimiser", "world_optimiser", "agent_optimiser",
                     "actor_optimiser", "critic_optimiser"):
            optimizer = getattr(self, name, None)
            if optimizer is not None:
                optimizer.defaults["foreach"] = bool(enabled)
                for group in optimizer.param_groups:
                    group["foreach"] = bool(enabled)
