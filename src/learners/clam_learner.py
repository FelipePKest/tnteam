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

    def train(self, batch, t_env, episode_num):
        clam_stats = None
        update_interval = getattr(self.args, "clam_update_interval", 1)
        if self.clam_updates % update_interval == 0:
            clam_stats = self._contrastive_update(batch)
        self.clam_updates += 1

        # PPO sees only the detached EMA representation.
        super().train(batch, t_env, episode_num)

        if clam_stats is not None and (
            t_env - self.clam_log_stats_t >= self.args.learner_log_interval
            or self.clam_log_stats_t == 0
        ):
            for key, value in clam_stats.items():
                self.logger.log_stat(key, value, t_env)
            self.clam_log_stats_t = t_env

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

    def _contrastive_update(self, batch):
        trajectories, lengths = self._select_ego_trajectories(batch)
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
        return {
            "clam_loss": loss.item(),
            "clam_grad_norm": grad_norm.item(),
            "clam_positive_cosine": cosine.item(),
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
