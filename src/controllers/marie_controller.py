import torch as th
from torch.distributions import Categorical
import os

from modules.marie_checkpoint import load_reference_marie_checkpoint
from modules.marie import MARIEPolicy


class MARIEMAC:
    """Fully controlled decentralized execution controller for MARIE."""

    def __init__(self, scheme, groups, args):
        self.args = args
        self.n_agents = args.n_agents
        self.policy = MARIEPolicy(
            scheme["obs"]["vshape"], args.n_agents, args.n_actions, args
        )
        self.action_selector = _MARIEActionSelector()

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        batch = ep_batch[bs]
        indices = list(range(self.n_agents))
        available = batch["avail_actions"][:, t_ep, indices].bool()
        logits, _ = self.policy.real_policy(batch, t_ep, indices, test_mode)
        logits = logits.masked_fill(~available, -1e9)
        if test_mode:
            # Upstream evaluates stochastically at temperature 1.0.
            actions = Categorical(logits=logits).sample()
        else:
            temperature = getattr(self.args, "marie_policy_temperature", 1.0)
            actions = Categorical(logits=logits / temperature).sample()
        hidden = batch["obs"].new_zeros(
            batch.batch_size, 1, self.n_agents, self.args.hidden_dim
        )
        return actions[:, None, :, None], hidden

    def forward(self, ep_batch, t=None, test_mode=False):
        if t is None:
            outputs = [
                self.policy.real_policy(
                    ep_batch, step, list(range(self.n_agents)), test_mode
                )[0][:, None]
                for step in range(ep_batch.max_seq_length)
            ]
            hidden = ep_batch["obs"].new_zeros(
                ep_batch.batch_size, ep_batch.max_seq_length,
                self.n_agents, self.args.hidden_dim,
            )
            return th.cat(outputs, 1), hidden
        logits, _ = self.policy.real_policy(
            ep_batch, t, list(range(self.n_agents)), test_mode
        )
        hidden = ep_batch["obs"].new_zeros(
            ep_batch.batch_size, 1, self.n_agents, self.args.hidden_dim
        )
        return logits[:, None], hidden

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
        native_path = os.path.join(path, "agent.th")
        if os.path.exists(native_path):
            self.policy.load_state_dict(th.load(native_path, map_location="cpu"))
            return
        reference_path = os.path.join(path, "reference_marie.pt")
        load_reference_marie_checkpoint(self.policy, reference_path)


class _MARIEActionSelector:
    pass
