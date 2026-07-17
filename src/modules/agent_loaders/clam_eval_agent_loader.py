import json
from types import SimpleNamespace as SN

import torch as th

from components.action_selectors import REGISTRY as action_REGISTRY
from modules.agent_loaders.base_agent_loader import BaseAgentLoader
from modules.agents import REGISTRY as agent_REGISTRY
from modules.clam import CLAMEncoder
from utils.load_utils import find_model_path


class CLAMEvalAgentLoader(BaseAgentLoader):
    """Load a trained CLAM actor and its EMA encoder for open evaluation."""

    def __init__(
        self,
        args,
        scheme,
        model_path,
        load_step,
        load_agent_idx,
        test_mode=True,
    ):
        del load_agent_idx  # CLAM uses parameter sharing.
        config_path = f"{model_path.replace('models', 'sacred')}/1/config.json"
        self.saved_args = SN(**json.load(open(config_path, "r")))
        super().__init__(
            args,
            scheme,
            n_agents=args.n_agents,
            obs_last_action=self.saved_args.obs_last_action,
            obs_agent_id=self.saved_args.obs_agent_id,
        )

        self.saved_args.n_agents = args.n_agents
        self.saved_args.n_actions = args.n_actions
        self.n_actions = args.n_actions
        self.agent_output_type = self.saved_args.agent_output_type
        self.action_selector = action_REGISTRY[
            self.saved_args.action_selector
        ](self.saved_args)
        if self.saved_args.agent != "rnn_clam":
            raise ValueError("CLAMEvalAgentLoader requires agent: rnn_clam")

        input_shape = self._get_input_shape(scheme, self.saved_args)
        model_dir, _ = find_model_path(
            model_path, load_step=load_step, logger=None
        )
        map_location = lambda storage, loc: storage
        target_encoder_state = th.load(
            f"{model_dir}/clam_target_encoder.th",
            map_location=map_location,
        )
        # Sacred records the pre-environment episode_limit (200 for MPE),
        # while the encoder is built after the environment reports its actual
        # limit (100). Reconstruct the positional table from the checkpoint.
        position_key = "position.encoding"
        if position_key in target_encoder_state:
            self.saved_args.episode_limit = (
                target_encoder_state[position_key].size(1) - 1
            )
        self.policy = agent_REGISTRY["rnn_clam"](input_shape, self.saved_args)
        self.policy.encoder = CLAMEncoder(
            obs_dim=scheme["obs"]["vshape"], args=self.saved_args
        )
        self.policy.load_state_dict(
            th.load(f"{model_dir}/agent.th", map_location=map_location)
        )
        # The actor is conditioned on the EMA target encoder during training.
        self.policy.encoder.load_state_dict(
            target_encoder_state
        )
        self.policy.encoder.eval()
        self.policy.eval()

        self.test_mode = test_mode
        self.device = args.device
        self.policy.to(self.device)

    def predict(
        self,
        ep_batch,
        agent_idx,
        t_ep,
        t_env,
        bs,
        test_mode=None,
    ):
        if test_mode is None:
            test_mode = self.test_mode

        ts = slice(None) if t_ep is None else slice(t_ep, t_ep + 1)
        batch = ep_batch[bs]
        agent_slice = slice(agent_idx, agent_idx + 1)
        available = batch["avail_actions"][:, ts, agent_slice]

        with th.no_grad():
            outputs, hidden = self.policy(batch, t=t_ep)
        outputs = outputs[:, :, agent_slice]
        hidden = hidden[:, :, agent_slice]

        if self.agent_output_type == "pi_logits" and getattr(
            self.saved_args, "mask_before_softmax", True
        ):
            outputs[available == 0] = -1e10

        actions = self.action_selector.select_action(
            outputs,
            available,
            t_env,
            test_mode=test_mode,
        )
        actions = actions.reshape(*outputs.shape[:-1], 1)
        return outputs, actions, hidden
