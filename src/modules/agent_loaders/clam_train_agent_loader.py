from components.action_selectors import REGISTRY as action_REGISTRY
from modules.agents import REGISTRY as agent_REGISTRY
from modules.agent_loaders.base_agent_loader import BaseAgentLoader


class CLAMTrainAgentLoader(BaseAgentLoader):
    """Expose the CLAM actor through the open-training controller."""

    def __init__(self, args, scheme, model_path):
        super().__init__(
            args,
            scheme,
            n_agents=args.n_agents,
            obs_last_action=args.obs_last_action,
            obs_agent_id=args.obs_agent_id,
        )
        if model_path:
            raise NotImplementedError(
                "Initialize CLAM through checkpoint_path, not agent_path"
            )
        input_shape = self._get_input_shape(scheme, args)
        self.policy = agent_REGISTRY[args.agent](input_shape, args)
        self.agent_output_type = args.agent_output_type
        self.action_selector = action_REGISTRY[args.action_selector](args)
        if args.agent != "rnn_clam":
            raise ValueError("CLAMTrainAgentLoader requires agent: rnn_clam")

    def predict(
        self,
        ep_batch,
        agent_idx_list,
        t_ep,
        t_env,
        bs,
        test_mode,
    ):
        ts = slice(None) if t_ep is None else slice(t_ep, t_ep + 1)
        batch = ep_batch[bs]
        available = batch["avail_actions"][:, ts, agent_idx_list]

        outputs, hidden = self.policy(batch, t=t_ep)
        outputs = outputs[:, :, agent_idx_list]
        hidden = hidden[:, :, agent_idx_list]

        if self.agent_output_type == "pi_logits" and getattr(
            self.args, "mask_before_softmax", True
        ):
            outputs[available == 0] = -1e10

        actions = self.action_selector.select_action(
            outputs,
            available,
            t_env,
            test_mode=test_mode,
        )
        return outputs, actions, hidden

    def set_encoder(self, encoder):
        self.policy.encoder = encoder
