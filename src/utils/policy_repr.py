import torch as th
import torch.nn as nn
import torch.nn.functional as F

from utils.mappo_util import init_module


class PolicyRepresentationHead(nn.Module):
    """
    Auxiliary head for learning policy representations.

    Given:
        - an agent observation o_t
        - an inferred policy embedding e_t

    It predicts:
        - the agent's current action a_t
        - optionally, the policy/type label z

    This replaces POAM's observation/action reconstruction decoder with
    a policy-behavior prediction objective.
    """

    def __init__(self, args, embed_dim, obs_dim, n_actions, n_policy_types=None):
        super().__init__()

        self.args = args
        self.embed_dim = embed_dim
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.n_policy_types = n_policy_types

        hidden_dim = getattr(args, "repr_hidden_dim", args.hidden_dim)

        self.action_head = nn.Sequential(
            init_module(nn.Linear(embed_dim + obs_dim, hidden_dim)),
            nn.ReLU(),
            init_module(nn.Linear(hidden_dim, hidden_dim)),
            nn.ReLU(),
            init_module(nn.Linear(hidden_dim, n_actions)),
        )

        if n_policy_types is not None and n_policy_types > 0:
            self.type_head = nn.Sequential(
                init_module(nn.Linear(embed_dim, hidden_dim)),
                nn.ReLU(),
                init_module(nn.Linear(hidden_dim, n_policy_types)),
            )
        else:
            self.type_head = None

    def forward_action(self, obs, embedding):
        """
        Args:
            obs:       Tensor with shape (..., obs_dim)
            embedding: Tensor with shape (..., embed_dim)

        Returns:
            action_logits: Tensor with shape (..., n_actions)
        """
        x = th.cat([obs, embedding], dim=-1)
        return self.action_head(x)

    def forward_type(self, embedding):
        """
        Args:
            embedding: Tensor with shape (..., embed_dim)

        Returns:
            type_logits: Tensor with shape (..., n_policy_types), or None
        """
        if self.type_head is None:
            return None

        return self.type_head(embedding)


def masked_cross_entropy(logits, targets, mask, n_classes, ignore_index=None):
    """
    Cross-entropy with an explicit mask.

    Args:
        logits: Tensor with shape (..., n_classes)
        targets: Tensor with shape (...)
        mask: Tensor with shape (...)
        n_classes: int
        ignore_index: optional int

    Returns:
        Scalar masked CE loss.
    """
    logits_flat = logits.reshape(-1, n_classes)
    targets_flat = targets.reshape(-1)
    mask_flat = mask.reshape(-1).float()

    if ignore_index is not None:
        valid = targets_flat != ignore_index
        mask_flat = mask_flat * valid.float()
        targets_flat = th.where(
            valid,
            targets_flat,
            th.zeros_like(targets_flat),
        )

    loss_flat = F.cross_entropy(
        logits_flat,
        targets_flat.long(),
        reduction="none",
    )

    return (loss_flat * mask_flat).sum() / (mask_flat.sum() + 1e-8)


def masked_accuracy(logits, targets, mask, ignore_index=None):
    preds = th.argmax(logits, dim=-1)
    targets_flat = targets.reshape(-1)
    preds_flat = preds.reshape(-1)
    mask_flat = mask.reshape(-1).float()

    if ignore_index is not None:
        valid = targets_flat != ignore_index
        mask_flat = mask_flat * valid.float()

    correct = (preds_flat == targets_flat).float() * mask_flat
    return correct.sum() / (mask_flat.sum() + 1e-8)


def supervised_contrastive_loss(embeddings, labels, mask, temperature=0.2, max_samples=2048):
    """
    Supervised contrastive loss over valid policy-type labels.

    Args:
        embeddings: Tensor with shape (bs, t, n_agents, embed_dim)
        labels: Tensor with shape (bs, t, n_agents), -1 for ignored samples
        mask: Tensor with shape (bs, t, n_agents)
    """
    emb = embeddings.reshape(-1, embeddings.shape[-1])
    labels = labels.reshape(-1).long()
    mask = mask.reshape(-1).float()
    valid = (mask > 0) & (labels >= 0)

    valid_idx = th.nonzero(valid, as_tuple=False).squeeze(-1)
    if valid_idx.numel() <= 1:
        return emb.sum() * 0.0

    if max_samples is not None and max_samples > 0 and valid_idx.numel() > max_samples:
        perm = th.randperm(valid_idx.numel(), device=embeddings.device)[:max_samples]
        valid_idx = valid_idx[perm]

    emb = F.normalize(emb[valid_idx], dim=-1)
    labels = labels[valid_idx]

    logits = th.matmul(emb, emb.transpose(0, 1)) / temperature
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()

    self_mask = th.eye(logits.shape[0], device=logits.device, dtype=th.bool)
    positive_mask = labels.unsqueeze(0).eq(labels.unsqueeze(1)) & (~self_mask)

    has_positive = positive_mask.any(dim=1)
    if not has_positive.any():
        return emb.sum() * 0.0

    exp_logits = th.exp(logits) * (~self_mask).float()
    log_prob = logits - th.log(exp_logits.sum(dim=1, keepdim=True) + 1e-8)
    mean_log_prob_pos = (positive_mask.float() * log_prob).sum(dim=1) / (
        positive_mask.float().sum(dim=1) + 1e-8
    )

    return -mean_log_prob_pos[has_positive].mean()
