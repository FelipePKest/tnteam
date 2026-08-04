import torch as th
from torch.distributions import Categorical

from modules.agent_loaders.base_agent_loader import BaseAgentLoader
from modules.matwm import MATWMPolicy


class MATWMTrainAgentLoader(BaseAgentLoader):
    """Makes MATWM's real-context policy available to OpenTrainMAC."""

    def __init__(self, args, scheme, model_path):
        super().__init__(
            args, scheme, n_agents=args.n_agents,
            obs_last_action=False, obs_agent_id=False,
            obs_team_composition=False,
        )
        self.policy = MATWMPolicy(
            scheme["obs"]["vshape"], args.n_agents, args.n_actions, args
        )
        self.action_selector = _MATWMActionSelector()
        if model_path:
            self.load_models(model_path)

    def predict(self, ep_batch, agent_idx_list, t_ep, t_env, bs, test_mode):
        batch = ep_batch[bs]
        available = batch["avail_actions"][:, t_ep, agent_idx_list].bool()
        if t_env < getattr(self.args, "matwm_prefill_steps", 1000) and not test_mode:
            actions = Categorical(available.float()).sample()
            logits = available.float().log().clamp_min(-1e9)
            hidden = batch["obs"].new_zeros(
                batch.batch_size, len(agent_idx_list), self.policy.hidden_dim
            )
        else:
            logits, _ = self.policy.real_policy(
                batch, t_ep, agent_idx_list, test_mode=test_mode
            )
            logits = logits.masked_fill(~available, -1e9)
            distribution = Categorical(logits=logits)
            actions = logits.argmax(-1) if test_mode else distribution.sample()
        runner_hidden = batch["obs"].new_zeros(
            batch.batch_size, len(agent_idx_list), self.args.hidden_dim
        )
        return logits[:, None], actions[:, None, :, None], runner_hidden[:, None]

    def init_hidden(self, batch_size):
        return next(self.policy.parameters()).new_zeros(
            batch_size, 1, self.n_agents, self.args.hidden_dim
        )


class _MATWMActionSelector:
    pass
