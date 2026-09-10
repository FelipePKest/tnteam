import json
import os
from types import SimpleNamespace

import torch as th
from torch.distributions import Categorical

from components.action_selectors import REGISTRY as action_REGISTRY
from modules.agent_loaders.base_agent_loader import BaseAgentLoader
from modules.marie import MARIEPolicy
from modules.marie_checkpoint import load_reference_marie_checkpoint
from utils.load_utils import find_model_path


class MARIETrainAgentLoader(BaseAgentLoader):
    """Expose the original MARIE policy through the open-team loader API."""

    def __init__(self, args, scheme, model_path=""):
        super().__init__(
            args, scheme, n_agents=args.n_agents,
            obs_last_action=False, obs_agent_id=False,
        )
        self.n_actions = args.n_actions
        self.policy = MARIEPolicy(
            scheme["obs"]["vshape"], args.n_agents, args.n_actions, args
        )
        self.action_selector = action_REGISTRY[args.action_selector](args)
        if model_path:
            self.load_models(model_path)

    def predict(
        self, ep_batch, agent_idx_list, t_ep, t_env, bs=slice(None),
        test_mode=False,
    ):
        batch = ep_batch[bs]
        indices = list(agent_idx_list)
        logits, _ = self.policy.real_policy(
            batch, t_ep, indices, test_mode=test_mode
        )
        available = batch["avail_actions"][:, t_ep, indices].bool()
        logits = logits.masked_fill(~available, -1e9)
        if not test_mode:
            temperature = getattr(
                self.args, "marie_policy_temperature", 1.0
            )
            logits = logits / temperature
        actions = Categorical(logits=logits).sample()
        hidden = batch["obs"].new_zeros(
            batch.batch_size, 1, len(indices), self.args.hidden_dim
        )
        return logits[:, None], actions[:, None, :, None], hidden

    def init_hidden(self, batch_size):
        return next(self.policy.parameters()).new_zeros(
            batch_size, 1, self.n_agents, self.args.hidden_dim
        )

    def load_models(self, path):
        native_path = os.path.join(path, "agent.th")
        if os.path.exists(native_path):
            return super().load_models(path)
        load_reference_marie_checkpoint(
            self.policy, os.path.join(path, "reference_marie.pt")
        )


class MARIEEvalAgentLoader(MARIETrainAgentLoader):
    """Load a saved MARIE checkpoint for stochastic paper-style evaluation."""

    def __init__(
        self, args, scheme, model_path, load_step="last", load_agent_idx=None,
        test_mode=True,
    ):
        del load_agent_idx
        config_path = os.path.join(
            model_path.replace("models", "sacred"), "1", "config.json"
        )
        policy_args = args
        if os.path.exists(config_path):
            with open(config_path, encoding="utf-8") as config_file:
                saved = json.load(config_file)
            saved.update({
                "n_agents": args.n_agents,
                "n_actions": args.n_actions,
                "device": args.device,
                "batch_size_run": args.batch_size_run,
            })
            policy_args = SimpleNamespace(**saved)
        super().__init__(policy_args, scheme)
        checkpoint, _ = find_model_path(
            model_path, load_step=load_step, logger=None
        )
        self.load_models(checkpoint)
        self.args = args
        self.test_mode = test_mode

    def predict(
        self, ep_batch, agent_idx, t_ep, t_env, bs=slice(None),
        test_mode=None,
    ):
        if test_mode is None:
            test_mode = self.test_mode
        indices = agent_idx if isinstance(agent_idx, (list, tuple)) else [agent_idx]
        return super().predict(
            ep_batch, indices, t_ep, t_env, bs=bs, test_mode=test_mode
        )
