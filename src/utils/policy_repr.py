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