import json
from types import SimpleNamespace as SN

import torch as th

from components.action_selectors import REGISTRY as action_REGISTRY
from modules.agent_loaders.base_agent_loader import BaseAgentLoader
from modules.agents import REGISTRY as agent_REGISTRY
from modules.clam import CLAMEncoder
from utils.load_utils import find_model_path


class CLAMEvalAgentLoader(BaseAgentLoader):
    """Load a frozen CLAM actor and its EMA trajectory encoder for evaluation."""

    preserve_policy_encoder = True

    def __init__(
        self,
        args,
        scheme,
        model_path,
        load_step,
        load_agent_idx=0,
        test_mode=True,
    ):
        del load_agent_idx  # CLAM uses one parameter-shared actor.
        config_path = f"{model_path.replace('models', 'sacred')}/1/config.json"
        with open(config_path, "r") as config_file:
            self.saved_args = SN(**json.load(config_file))

        super().__init__(
            args,
            scheme,
            n_agents=args.n_agents,
            obs_last_action=self.saved_args.obs_last_action,
            obs_agent_id=self.saved_args.obs_agent_id,
        )
        if self.saved_args.agent != "rnn_clam":
            raise ValueError(
                f"CLAMEvalAgentLoader requires agent=rnn_clam, got {self.saved_args.agent}"
            )

        self.saved_args.n_agents = args.n_agents
        self.saved_args.n_actions = args.n_actions
        self.saved_args.device = args.device
        checkpoint, _ = find_model_path(model_path, load_step=load_step, logger=None)
        map_location = lambda storage, loc: storage
        actor_state = th.load(f"{checkpoint}/agent.th", map_location=map_location)
        encoder_state = th.load(
            f"{checkpoint}/clam_target_encoder.th",
            map_location=map_location,
        )
        # Sacred captures the config before the runner adds episode_limit.
        # Recover the exact training horizon from CLAM's positional buffer.
        position_key = "position.encoding"
        if position_key in encoder_state:
            self.saved_args.episode_limit = encoder_state[position_key].shape[1] - 1
        self.agent_output_type = self.saved_args.agent_output_type
        self.action_selector = action_REGISTRY[self.saved_args.action_selector](
            self.saved_args
        )
        input_shape = self._get_input_shape(scheme, self.saved_args)
        self.policy = agent_REGISTRY["rnn_clam"](input_shape, self.saved_args)

        obs_shape = scheme["obs"]["vshape"]
        obs_dim = obs_shape if isinstance(obs_shape, int) else obs_shape[0]
        self.policy.encoder = CLAMEncoder(obs_dim=obs_dim, args=self.saved_args)

        self.policy.load_state_dict(actor_state)
        self.policy.encoder.load_state_dict(encoder_state)
        self.policy.encoder.eval()
        for parameter in self.policy.parameters():
            parameter.requires_grad_(False)

        self.use_param_sharing = True
        self.batch_size_run = args.batch_size_run
        self.device = args.device
        self.test_mode = test_mode
        self.policy.to(self.device)
        self.policy.eval()

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

        batch = ep_batch[bs]
        ts = slice(None) if t_ep is None else slice(t_ep, t_ep + 1)
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

    def cuda(self):
        self.policy.to(self.device)
