import torch as th
import torch.nn as nn

from utils.mappo_util import init_module, init_rnn
from utils.mlp import MLPBase
from utils.popart import PopArt


class CLAMCritic(nn.Module):
    """Decentralized value function conditioned on the CLAM context."""

    def __init__(self, scheme, args):
        super().__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.input_size = self._get_input_shape(scheme)
        self.encoder = None

        self.base = MLPBase(
            self.input_size + args.embed_dim,
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
            self.rnn = MLPBase(
                args.hidden_dim,
                args.hidden_dim,
                n_hidden_layers=0,
                use_feature_norm=args.use_obs_norm,
                use_orthogonal=args.use_orthogonal_init,
            )

        if args.use_popart:
            self.v_out = init_module(
                PopArt(args.hidden_dim, 1, norm_axes=3, device=args.device),
                gain=1.0,
            )
        else:
            self.v_out = init_module(
                nn.Linear(args.hidden_dim, 1, device=args.device), gain=1.0
            )
        self.value_normalizer = self.v_out if args.use_popart else None

    def init_hidden(self):
        return self.base.mlp.fc1[0].weight.new(
            self.args.batch_size,
            1,
            self.n_agents,
            self.args.hidden_dim,
        ).zero_()

    def forward(self, batch, hidden_state, t=None, build_inputs=True):
        if self.encoder is None:
            raise RuntimeError("CLAM encoder has not been attached to the critic")
        inputs = self._build_inputs(batch, t=t) if build_inputs else batch
        original_dims = inputs.shape[:-1]

        valid = None
        if build_inputs and "filled" in batch.data.transition_data:
            valid = batch["filled"].squeeze(-1)

        with th.no_grad():
            if not build_inputs:
                raise ValueError("CLAMCritic requires an EpisodeBatch to build context")
            if t is None:
                contexts = self.encoder.forward_prefixes(batch["obs"], valid=valid)
            else:
                observations = batch["obs"][:, : t + 1]
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

        inputs = th.cat([inputs, contexts.detach()], dim=-1)
        flat_inputs = inputs.reshape(-1, self.input_size + self.args.embed_dim)
        flat_hidden = hidden_state.reshape(-1, self.args.hidden_dim)

        x = self.base(flat_inputs)
        if self.args.use_rnn:
            h_out = self.rnn(x, flat_hidden)
            h_norm = self.rnn_norm(h_out)
        else:
            h_norm = h_out = self.rnn(x)
        values = self.v_out(h_norm)
        return values.view(*original_dims, -1), h_out.view(*original_dims, -1)

    def _build_inputs(self, batch, t=None):
        batch_size = batch.batch_size
        max_t = batch.max_seq_length if t is None else 1
        ts = slice(None) if t is None else slice(t, t + 1)

        inputs = [batch["obs"][:, ts]]
        if self.args.obs_state:
            inputs.append(
                batch["state"][:, ts]
                .unsqueeze(-2)
                .expand(-1, -1, self.n_agents, -1)
            )
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
            inputs.append(
                th.eye(self.n_agents, device=batch.device)
                .unsqueeze(0)
                .unsqueeze(0)
                .expand(batch_size, max_t, -1, -1)
            )
        return th.cat(inputs, dim=-1)

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_state:
            input_shape += scheme["state"]["vshape"]
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
        if self.args.obs_agent_id:
            input_shape += self.n_agents
        return input_shape
