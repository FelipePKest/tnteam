import torch as th
import torch.nn as nn
import torch.nn.functional as F

from utils.mappo_util import init_module, init_rnn
from utils.mlp import MLPBase


class RNNCLAMAgent(nn.Module):
    """PPO actor conditioned on CLAM's real-time policy representation."""

    def __init__(self, input_shape, args):
        super().__init__()
        self.args = args
        self.input_size = input_shape
        self.n_agents = args.n_agents
        # The learner attaches its EMA target encoder after constructing the
        # actor optimiser, so CLAM parameters are trained only by InfoNCE.
        self.encoder = None

        self.base = MLPBase(
            input_shape + args.embed_dim,
            args.hidden_dim,
            n_hidden_layers=1,
            use_feature_norm=args.use_obs_norm,
            use_orthogonal=args.use_orthogonal_init,
        )
        if args.use_rnn:
            self.rnn = init_rnn(
                nn.GRUCell(args.hidden_dim, args.hidden_dim),
                args.use_orthogonal_init,
            )
            self.rnn_norm = nn.LayerNorm(args.hidden_dim)
        else:
            self.rnn = init_module(nn.Linear(args.hidden_dim, args.hidden_dim))
        self.fc2 = init_module(nn.Linear(args.hidden_dim, args.n_actions))

    def init_hidden(self, batch_size):
        return self.base.mlp.fc1[0].weight.new(
            batch_size, 1, self.n_agents, self.args.hidden_dim
        ).zero_()

    def forward(self, ep_batch, t=None, hidden_state=None):
        if self.encoder is None:
            raise RuntimeError("CLAM encoder has not been attached to the actor")

        ts = slice(None) if t is None else slice(t, t + 1)
        inputs = self._build_inputs(ep_batch, t)
        original_dims = inputs.shape[:-1]

        if hidden_state is None:
            hidden_state = ep_batch["actor_hidden_states"][:, ts]

        valid = None
        if "filled" in ep_batch.data.transition_data:
            valid = ep_batch["filled"].squeeze(-1)

        with th.no_grad():
            if hasattr(ep_batch, "clam_contexts"):
                contexts = ep_batch.clam_contexts[:, ts]
            elif t is None:
                contexts = self.encoder.forward_prefixes(
                    ep_batch["obs"], valid=valid
                )
            else:
                observations = ep_batch["obs"][:, : t + 1]
                batch_size, _, n_agents, obs_dim = observations.shape
                trajectories = observations.permute(0, 2, 1, 3).reshape(
                    batch_size * n_agents, t + 1, obs_dim
                )
                padding_mask = None
                if valid is not None:
                    padding_mask = (~valid[:, : t + 1].bool()).unsqueeze(1)
                    padding_mask = padding_mask.expand(-1, n_agents, -1).reshape(
                        batch_size * n_agents, t + 1
                    )
                contexts = self.encoder(
                    trajectories, padding_mask=padding_mask
                ).view(batch_size, 1, n_agents, -1)

        if getattr(self.args, "clam_zero_context", False):
            contexts = th.zeros_like(contexts)
        inputs = th.cat([inputs, contexts.detach()], dim=-1)
        flat_inputs = inputs.reshape(-1, self.input_size + self.args.embed_dim)
        flat_hidden = hidden_state.reshape(-1, self.args.hidden_dim)

        x = self.base(flat_inputs)
        if self.args.use_rnn:
            h_out = self.rnn(x, flat_hidden)
            h_norm = self.rnn_norm(h_out)
        else:
            h_norm = h_out = F.relu(self.rnn(x))

        logits = self.fc2(h_norm)
        return (
            logits.view(*original_dims, -1),
            h_out.view(*original_dims, -1),
        )

    def _build_inputs(self, batch, t=None):
        batch_size = batch.batch_size
        max_t = batch.max_seq_length if t is None else 1
        ts = slice(None) if t is None else slice(t, t + 1)

        inputs = [batch["obs"][:, ts]]
        if self.args.obs_last_action:
            if t is None:
                last_actions = th.cat(
                    [
                        th.zeros_like(batch["actions_onehot"][:, [0]]),
                        batch["actions_onehot"][:, : max_t - 1],
                    ],
                    dim=1,
                )
            elif t == 0:
                last_actions = th.zeros_like(batch["actions_onehot"][:, ts])
            else:
                last_actions = batch["actions_onehot"][:, t - 1 : t]
            inputs.append(last_actions)

        if self.args.obs_agent_id:
            agent_ids = th.eye(self.n_agents, device=batch.device).expand(
                batch_size, max_t, self.n_agents, -1
            )
            inputs.append(agent_ids)
        return th.cat(inputs, dim=-1)
