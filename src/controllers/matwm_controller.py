import torch as th
from torch.distributions import Categorical

from modules.matwm import MATWMPolicy


class MATWMMAC:
    """Controller used by MATWM in fully controlled cooperative environments."""

    def __init__(self, scheme, groups, args):
        self.args = args
        self.n_agents = args.n_agents
        self.policy = MATWMPolicy(
            scheme["obs"]["vshape"], args.n_agents, args.n_actions, args
        )
        # Runner logging expects this attribute to exist.
        self.action_selector = _MATWMActionSelector()

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        batch = ep_batch[bs]
        indices = list(range(self.n_agents))
        available = batch["avail_actions"][:, t_ep, indices].bool()
        if t_env < getattr(self.args, "matwm_prefill_steps", 1000) and not test_mode:
            actions = Categorical(available.float()).sample()
            hidden = batch["obs"].new_zeros(
                batch.batch_size, self.n_agents, self.policy.hidden_dim
            )
        else:
            logits, _ = self.policy.real_policy(batch, t_ep, indices, test_mode)
            logits = logits.masked_fill(~available, -1e9)
            distribution = Categorical(logits=logits)
            actions = logits.argmax(-1) if test_mode else distribution.sample()
        runner_hidden = batch["obs"].new_zeros(
            batch.batch_size, self.n_agents, self.args.hidden_dim
        )
        return actions[:, None, :, None], runner_hidden[:, None]

    def forward(self, ep_batch, t=None, test_mode=False):
        if t is None:
            outputs = []
            hidden = []
            for step in range(ep_batch.max_seq_length):
                logits, h = self.policy.real_policy(
                    ep_batch, step, list(range(self.n_agents)), test_mode
                )
                outputs.append(logits[:, None])
                hidden.append(ep_batch["obs"].new_zeros(
                    ep_batch.batch_size, 1, self.n_agents, self.args.hidden_dim
                ))
            return th.cat(outputs, 1), th.cat(hidden, 1)
        logits, _ = self.policy.real_policy(
            ep_batch, t, list(range(self.n_agents)), test_mode
        )
        runner_hidden = ep_batch["obs"].new_zeros(
            ep_batch.batch_size, 1, self.n_agents, self.args.hidden_dim
        )
        return logits[:, None], runner_hidden

    def init_hidden(self, batch_size):
        return next(self.policy.parameters()).new_zeros(
            batch_size, 1, self.n_agents, self.args.hidden_dim
        )

    def parameters(self):
        return self.policy.parameters()

    def cuda(self):
        self.policy.cuda()

    def save_models(self, path):
        th.save(self.policy.state_dict(), f"{path}/agent.th")

    def load_models(self, path):
        self.policy.load_state_dict(th.load(f"{path}/agent.th", map_location="cpu"))


class _MATWMActionSelector:
    """Compatibility shim for the runner's optional epsilon logging."""

    pass
