import copy
import os

import torch as th
from torch.optim import Adam

from controllers import AgentOwnedMAC, OpenTrainMAC
from learners.ppo_learner import PPOLearner
from modules.clam import CLAMEncoder, CLAMProjectionHead, symmetric_info_nce


class CLAMLearner(PPOLearner):
    """PPO with Contrastive Learning-based Agent Modeling (CLAM).

    The actor and critic consume representations from an EMA target encoder.
    The online encoder and projection head are updated solely by asymmetric
    trajectory-view InfoNCE, as described in Ma et al. (2025).
    """

    def __init__(self, mac, scheme, logger, args):
        # PPOLearner deliberately constructs the PPO optimisers before CLAM is
        # attached. This prevents PPO gradients from updating the encoder.
        super().__init__(mac, scheme, logger, args)

        obs_dim = scheme["obs"]["vshape"]
        self.clam_encoder = CLAMEncoder(obs_dim=obs_dim, args=args)
        self.target_clam_encoder = copy.deepcopy(self.clam_encoder)
        self.clam_projector = CLAMProjectionHead(args.embed_dim, args)
        init_path = getattr(args, "clam_init_encoder_path", "")
        if init_path:
            if os.path.isfile(init_path):
                encoder_path = init_path
                target_path = os.path.join(
                    os.path.dirname(init_path), "clam_target_encoder.th"
                )
            else:
                encoder_path = os.path.join(init_path, "clam_encoder.th")
                target_path = os.path.join(
                    init_path, "clam_target_encoder.th"
                )
            map_location = lambda storage, loc: storage
            self.clam_encoder.load_state_dict(
                th.load(encoder_path, map_location=map_location)
            )
            if os.path.exists(target_path):
                self.target_clam_encoder.load_state_dict(
                    th.load(target_path, map_location=map_location)
                )
            else:
                self.target_clam_encoder.load_state_dict(
                    self.clam_encoder.state_dict()
                )
        self.clam_params = list(self.clam_encoder.parameters()) + list(
            self.clam_projector.parameters()
        )
        self.clam_optimiser = Adam(
            self.clam_params,
            lr=args.clam_lr,
            eps=args.optim_eps,
        )

        for parameter in self.target_clam_encoder.parameters():
            parameter.requires_grad_(False)
        self.target_clam_encoder.eval()

        if isinstance(mac, AgentOwnedMAC):
            mac.agent.encoder = self.target_clam_encoder
        elif isinstance(mac, OpenTrainMAC):
            mac.set_encoder(self.target_clam_encoder)
        else:
            raise TypeError(
                "CLAM requires agent_owned_mac or open_train_mac, got "
                f"{type(mac).__name__}"
            )
        self.critic.encoder = self.target_clam_encoder

        self.clam_updates = 0
        self.clam_log_stats_t = 0
        self.clam_buffer_capacity = getattr(
            args, "clam_buffer_capacity", 10000
        )
        self.clam_batch_size = getattr(args, "clam_batch_size", 512)
        if self.clam_buffer_capacity < self.clam_batch_size:
            raise ValueError(
                "clam_buffer_capacity must be at least clam_batch_size"
            )
        max_length = getattr(args, "episode_limit", 200) + 1
        self.clam_trajectory_buffer = th.zeros(
            self.clam_buffer_capacity,
            max_length,
            obs_dim,
            dtype=th.float32,
            device="cpu",
        )
        self.clam_length_buffer = th.zeros(
            self.clam_buffer_capacity, dtype=th.long, device="cpu"
        )
        self.clam_label_buffer = th.full(
            (self.clam_buffer_capacity,), -1, dtype=th.long, device="cpu"
        )
        self.clam_buffer_index = 0
        self.clam_buffer_count = 0

    def train(self, batch, t_env, episode_num):
        # PPO must be updated before the EMA context encoder changes. Otherwise
        # its recomputed "old" log-probabilities do not correspond to the
        # context used to collect the rollout.
        self._cache_target_contexts(batch)
        super().train(batch, t_env, episode_num)
        self._store_clam_batch(batch)

        clam_stats = None
        update_interval = getattr(self.args, "clam_update_interval", 1)
        disable_updates = (
            getattr(self.args, "clam_zero_context", False)
            or getattr(self.args, "clam_freeze_encoder", False)
        )
        can_sample = self.clam_buffer_count >= self.clam_batch_size
        if (
            not disable_updates
            and can_sample
            and self.clam_updates % update_interval == 0
        ):
            trajectories, lengths, labels = self._sample_clam_batch()
            clam_stats = self._contrastive_update(
                trajectories, lengths, labels
            )
        self.clam_updates += 1

        if clam_stats is not None and (
            t_env - self.clam_log_stats_t >= self.args.learner_log_interval
            or self.clam_log_stats_t == 0
        ):
            clam_stats["clam_buffer_size"] = float(self.clam_buffer_count)
            for key, value in clam_stats.items():
                self.logger.log_stat(key, value, t_env)
            self.clam_log_stats_t = t_env

    @th.no_grad()
    def _cache_target_contexts(self, batch):
        if getattr(self.args, "clam_zero_context", False):
            batch.clam_contexts = batch["obs"].new_zeros(
                *batch["obs"].shape[:-1], self.args.embed_dim
            )
            return
        valid = batch["filled"].squeeze(-1)
        contexts = self.target_clam_encoder.forward_prefixes(
            batch["obs"], valid=valid
        )
        batch.clam_contexts = contexts.detach()

    def _get_clam_labels(self, batch):
        return th.full(
            (batch.batch_size,), -1, dtype=th.long, device=batch.device
        )

    @th.no_grad()
    def _store_clam_batch(self, batch):
        trajectories, lengths = self._select_ego_trajectories(batch)
        labels = self._get_clam_labels(batch)
        trajectories = trajectories.detach().to("cpu")
        lengths = lengths.detach().to("cpu")
        labels = labels.detach().to("cpu")

        batch_size = trajectories.size(0)
        if batch_size >= self.clam_buffer_capacity:
            trajectories = trajectories[-self.clam_buffer_capacity :]
            lengths = lengths[-self.clam_buffer_capacity :]
            labels = labels[-self.clam_buffer_capacity :]
            batch_size = self.clam_buffer_capacity

        first = min(
            batch_size, self.clam_buffer_capacity - self.clam_buffer_index
        )
        second = batch_size - first
        slices = (
            (slice(self.clam_buffer_index, self.clam_buffer_index + first),
             slice(0, first)),
        )
        if second:
            slices += ((slice(0, second), slice(first, batch_size)),)

        for destination, source in slices:
            self.clam_trajectory_buffer[destination].zero_()
            sequence_length = min(
                trajectories.size(1),
                self.clam_trajectory_buffer.size(1),
            )
            self.clam_trajectory_buffer[destination, :sequence_length].copy_(
                trajectories[source, :sequence_length]
            )
            self.clam_length_buffer[destination].copy_(
                lengths[source].clamp(max=sequence_length)
            )
            self.clam_label_buffer[destination].copy_(labels[source])

        self.clam_buffer_index = (
            self.clam_buffer_index + batch_size
        ) % self.clam_buffer_capacity
        self.clam_buffer_count = min(
            self.clam_buffer_capacity, self.clam_buffer_count + batch_size
        )

    def _sample_clam_batch(self):
        indices = th.randperm(self.clam_buffer_count)[: self.clam_batch_size]
        device = self.args.device
        return (
            self.clam_trajectory_buffer[indices].to(device),
            self.clam_length_buffer[indices].to(device),
            self.clam_label_buffer[indices].to(device),
        )

    def _select_ego_trajectories(self, batch):
        """Choose one local ego trajectory per episode.

        Keeping one sample per episode preserves the paper's definition of
        positives (two views of one episodic trajectory) and avoids treating
        different agents from the same episode as contrastive negatives.
        """
        observations = batch["obs"]
        batch_size = observations.size(0)
        device = observations.device

        if self.args.open_train_or_eval:
            trainable = batch["trainable_agents"][:, 0, :, 0].bool().clone()
            # There should always be at least one controlled agent. Fall back
            # to all agents for malformed/debug batches instead of crashing.
            missing = ~trainable.any(dim=1)
            trainable[missing] = True
            agent_indices = th.multinomial(trainable.float(), 1).squeeze(1)
        else:
            agent_indices = th.randint(
                self.n_agents, (batch_size,), device=device
            )

        trajectories = observations[
            th.arange(batch_size, device=device), :, agent_indices
        ]
        lengths = batch["filled"].squeeze(-1).sum(dim=1).long()
        lengths = lengths.clamp(min=1, max=observations.size(1))
        return trajectories, lengths

    def _random_crop(self, trajectories, lengths, crop_length):
        crops = []
        for trajectory, length in zip(trajectories, lengths):
            max_start = int(length.item()) - crop_length
            start = int(
                th.randint(
                    max_start + 1,
                    (1,),
                    device=trajectories.device,
                ).item()
            )
            crops.append(trajectory[start : start + crop_length])
        return th.stack(crops, dim=0)

    def _strong_augmentation(self, trajectories):
        ratio = self.args.clam_mask_ratio
        if ratio <= 0:
            return trajectories, 0.0

        augmented = trajectories.clone()
        n_masked = min(
            trajectories.size(1) - 1,
            max(1, int(round(trajectories.size(1) * ratio))),
        )
        for row in range(trajectories.size(0)):
            indices = th.randperm(
                trajectories.size(1), device=trajectories.device
            )[:n_masked]
            augmented[row, indices] = 0
        return augmented, n_masked / trajectories.size(1)

    def _contrastive_update(self, trajectories, lengths, labels=None):
        eligible = lengths >= 2
        trajectories = trajectories[eligible]
        lengths = lengths[eligible]
        if trajectories.size(0) < 2:
            return None

        shortest = int(lengths.min().item())
        configured_min = getattr(self.args, "clam_min_crop", 8)
        configured_max = getattr(
            self.args, "clam_max_crop", self.args.episode_limit
        )
        lower = min(configured_min, shortest)
        upper = min(configured_max, shortest)
        lower = max(2, lower)
        upper = max(lower, upper)

        first_length = int(
            th.randint(lower, upper + 1, (1,), device=trajectories.device).item()
        )
        second_length = int(
            th.randint(lower, upper + 1, (1,), device=trajectories.device).item()
        )
        strong_view = self._random_crop(
            trajectories, lengths, first_length
        )
        weak_view = self._random_crop(
            trajectories, lengths, second_length
        )
        strong_view, masked_fraction = self._strong_augmentation(strong_view)

        self.clam_encoder.train()
        first_context = self.clam_encoder(strong_view)
        second_context = self.clam_encoder(weak_view)
        first_projection = self.clam_projector(first_context)
        second_projection = self.clam_projector(second_context)
        loss = symmetric_info_nce(
            first_projection,
            second_projection,
            self.args.clam_temperature,
        )

        self.clam_optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(
            self.clam_params, self.args.clam_grad_norm_clip
        )
        self.clam_optimiser.step()
        self._momentum_update_target()

        cosine = (first_projection * second_projection).sum(dim=-1).mean()
        cross_similarity = first_projection @ second_projection.transpose(0, 1)
        negative_mask = ~th.eye(
            cross_similarity.size(0),
            dtype=th.bool,
            device=cross_similarity.device,
        )
        negative_cosine = cross_similarity[negative_mask].mean()
        representation_std = th.cat(
            [first_context, second_context], dim=0
        ).std(dim=0).mean()
        centered = th.cat([first_context, second_context], dim=0)
        centered = centered - centered.mean(dim=0, keepdim=True)
        singular_values = th.linalg.svdvals(centered)
        probabilities = singular_values / singular_values.sum().clamp_min(1e-12)
        effective_rank = th.exp(
            -(probabilities * probabilities.clamp_min(1e-12).log()).sum()
        )
        return {
            "clam_loss": loss.item(),
            "clam_grad_norm": grad_norm.item(),
            "clam_positive_cosine": cosine.item(),
            "clam_negative_cosine": negative_cosine.item(),
            "clam_cosine_margin": (cosine - negative_cosine).item(),
            "clam_representation_std": representation_std.item(),
            "clam_effective_rank": effective_rank.item(),
            "clam_strong_crop_length": first_length,
            "clam_weak_crop_length": second_length,
            "clam_masked_fraction": masked_fraction,
        }

    @th.no_grad()
    def _momentum_update_target(self):
        tau = self.args.clam_target_tau
        for online, target in zip(
            self.clam_encoder.parameters(),
            self.target_clam_encoder.parameters(),
        ):
            target.data.mul_(1.0 - tau).add_(online.data, alpha=tau)
        # Copy buffers (notably positional encodings) exactly.
        for online, target in zip(
            self.clam_encoder.buffers(),
            self.target_clam_encoder.buffers(),
        ):
            target.copy_(online)
        self.target_clam_encoder.eval()

    def cuda(self):
        super().cuda()
        self.clam_encoder.cuda()
        self.target_clam_encoder.cuda()
        self.clam_projector.cuda()

    def save_models(self, path):
        super().save_models(path)
        th.save(self.clam_encoder.state_dict(), os.path.join(path, "clam_encoder.th"))
        th.save(
            self.target_clam_encoder.state_dict(),
            os.path.join(path, "clam_target_encoder.th"),
        )
        th.save(
            self.clam_projector.state_dict(),
            os.path.join(path, "clam_projector.th"),
        )
        th.save(
            self.clam_optimiser.state_dict(),
            os.path.join(path, "clam_opt.th"),
        )

    def load_models(self, path):
        super().load_models(path)
        map_location = lambda storage, loc: storage
        self.clam_encoder.load_state_dict(
            th.load(
                os.path.join(path, "clam_encoder.th"),
                map_location=map_location,
            )
        )
        self.target_clam_encoder.load_state_dict(
            th.load(
                os.path.join(path, "clam_target_encoder.th"),
                map_location=map_location,
            )
        )
        self.clam_projector.load_state_dict(
            th.load(
                os.path.join(path, "clam_projector.th"),
                map_location=map_location,
            )
        )
        self.clam_optimiser.load_state_dict(
            th.load(os.path.join(path, "clam_opt.th"), map_location=map_location)
        )
        self.target_clam_encoder.eval()
