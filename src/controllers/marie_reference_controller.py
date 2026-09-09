import numpy as np
import torch as th

from marie_reference_service import MARIEReferenceService


class MARIEReferenceMAC:
    """EPyMARL controller facade backed by upstream MARIE."""

    def __init__(self, scheme, groups, args):
        if args.batch_size_run != 1:
            raise ValueError("Reference MARIE requires batch_size_run=1")
        self.args = args
        self.n_agents = args.n_agents
        self.action_selector = _Selector()
        self.service = MARIEReferenceService(args, scheme["obs"]["vshape"])

    def init_hidden(self, batch_size):
        self.service.reset(batch_size)
        return th.zeros(batch_size, 1, self.n_agents, self.args.hidden_dim)

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        obs = ep_batch[bs]["obs"][:, t_ep].detach().cpu().numpy()
        avail = ep_batch[bs]["avail_actions"][:, t_ep].detach().cpu().numpy()
        actions, _ = self.service.act(obs, avail)
        # DreamerController removes its singleton rollout batch dimension and
        # returns [n_agents]. EPyMARL keeps that dimension in its MAC contract.
        result = th.as_tensor(
            actions, dtype=th.long, device=ep_batch.device
        ).reshape(1, self.n_agents)
        hidden = th.zeros(1, 1, self.n_agents, self.args.hidden_dim, device=ep_batch.device)
        return result[:, None, :, None], hidden

    def cuda(self):
        pass

    def save_models(self, path):
        self.service.save(path)

    def load_models(self, path):
        self.service.load(path)


class _Selector:
    pass
