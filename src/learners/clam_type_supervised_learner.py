import torch as th

from learners.clam_learner import CLAMLearner
from modules.clam import supervised_contrastive_loss


class TypeSupervisedCLAMLearner(CLAMLearner):
    """CLAM variant supervised by the sampled uncontrolled-agent type.

    Unlike the original CLAM objective, which treats only two views from the
    same episode as positives, this learner treats every view from episodes
    with the same ``uncontrolled_team_idx`` as positive. Views collected
    against different uncontrolled teams are negatives.
    """

    def _get_uncontrolled_team_labels(self, batch):
        if not self.args.open_train_or_eval:
            raise ValueError(
                "Type-supervised CLAM requires open training and "
                "uncontrolled_team_idx labels"
            )
        if "uncontrolled_team_idx" not in batch.data.episode_data:
            raise ValueError(
                "CLAM batch is missing the episode-level uncontrolled_team_idx"
            )
        labels = batch["uncontrolled_team_idx"]
        return labels.reshape(labels.size(0), -1)[:, 0].long()

    def _contrastive_update(self, batch):
        trajectories, lengths = self._select_ego_trajectories(batch)
        labels = self._get_uncontrolled_team_labels(batch)
        eligible = (lengths >= 2) & (labels >= 0)
        trajectories = trajectories[eligible]
        lengths = lengths[eligible]
        labels = labels[eligible]

        unique_labels, label_counts = th.unique(labels, return_counts=True)
        if trajectories.size(0) < 2 or unique_labels.numel() < 2:
            # A type-supervised update needs at least one cross-type negative.
            return {
                "clam_update_skipped": 1.0,
                "clam_num_types": float(unique_labels.numel()),
            }

        shortest = int(lengths.min().item())
        configured_min = getattr(self.args, "clam_min_crop", 8)
        configured_max = getattr(
            self.args, "clam_max_crop", self.args.episode_limit
        )
        lower = max(2, min(configured_min, shortest))
        upper = max(lower, min(configured_max, shortest))

        first_length = int(
            th.randint(lower, upper + 1, (1,), device=trajectories.device).item()
        )
        second_length = int(
            th.randint(lower, upper + 1, (1,), device=trajectories.device).item()
        )
        strong_view = self._random_crop(trajectories, lengths, first_length)
        weak_view = self._random_crop(trajectories, lengths, second_length)
        strong_view, masked_fraction = self._strong_augmentation(strong_view)

        self.clam_encoder.train()
        first_context = self.clam_encoder(strong_view)
        second_context = self.clam_encoder(weak_view)
        first_projection = self.clam_projector(first_context)
        second_projection = self.clam_projector(second_context)
        loss = supervised_contrastive_loss(
            first_projection,
            second_projection,
            labels,
            self.args.clam_temperature,
        )

        self.clam_optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(
            self.clam_params, self.args.clam_grad_norm_clip
        )
        self.clam_optimiser.step()
        self._momentum_update_target()

        with th.no_grad():
            projected = th.cat([first_projection, second_projection], dim=0)
            projected_labels = labels.repeat(2)
            similarities = projected @ projected.transpose(0, 1)
            unique_pairs = th.triu(
                th.ones_like(similarities, dtype=th.bool), diagonal=1
            )
            same_type = (
                projected_labels[:, None].eq(projected_labels[None, :])
                & unique_pairs
            )
            different_type = (
                projected_labels[:, None].ne(projected_labels[None, :])
                & unique_pairs
            )
            same_type_cosine = similarities[same_type].mean()
            different_type_cosine = similarities[different_type].mean()

        return {
            "clam_loss": loss.item(),
            "clam_grad_norm": grad_norm.item(),
            "clam_same_type_cosine": same_type_cosine.item(),
            "clam_different_type_cosine": different_type_cosine.item(),
            "clam_type_cosine_margin": (
                same_type_cosine - different_type_cosine
            ).item(),
            "clam_strong_crop_length": first_length,
            "clam_weak_crop_length": second_length,
            "clam_masked_fraction": masked_fraction,
            "clam_update_skipped": 0.0,
            "clam_num_types": float(unique_labels.numel()),
            "clam_min_episodes_per_type": float(label_counts.min().item()),
            "clam_max_episodes_per_type": float(label_counts.max().item()),
        }
