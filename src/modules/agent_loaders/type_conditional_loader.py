import os
from collections import defaultdict
from typing import Dict, List, Sequence

import torch as th

from modules.agent_loaders import REGISTRY as agent_loader_REGISTRY
from modules.classifiers import UncontrolledTransformerClassifier


class TypeConditionalAgentLoader:
    """
    Wrapper that hosts a teammate-type classifier and a pool of expert policies.
    For every predict() call we classify each environment instance and dispatch
    the query to the corresponding expert policy, reassembling the outputs into
    the original batch order.
    """

    def __init__(
        self,
        args,
        scheme,
        classifier_cfg: Dict,
        teammate_cfgs: Sequence[Dict],
        base_uncntrl_path: str,
    ):
        self.args = args
        self.scheme = scheme
        self.device = args.device
        self.history_len = classifier_cfg.get("history_len", 32)

        obs_shape = scheme["obs"]["vshape"]
        if isinstance(obs_shape, int):
            obs_dim = obs_shape
        else:
            obs_dim = 1
            for dim in obs_shape:
                obs_dim *= dim

        self.classifier = UncontrolledTransformerClassifier(
            obs_dim=obs_dim,
            n_agents=args.n_agents,
            episode_limit=args.episode_limit,
            num_uncontrolled_types=len(teammate_cfgs),
            d_model=classifier_cfg.get("d_model", 128),
            nhead=classifier_cfg.get("nhead", 4),
            num_layers=classifier_cfg.get("num_layers", 2),
            dim_feedforward=classifier_cfg.get("ff", 256),
            dropout=classifier_cfg.get("dropout", 0.1),
        ).to(self.device)

        ckpt_path = classifier_cfg.get("checkpoint", "")
        if ckpt_path:
            state = th.load(ckpt_path, map_location=self.device)
            self.classifier.load_state_dict(state)

        self.type_names: List[str] = []
        self.type_name_to_idx: Dict[str, int] = {}
        self.label_mapping: Dict[str, int] = {}
        self.acc_correct = 0
        self.acc_total = 0
        self.experts: Dict[str, object] = {}
        for type_cfg in teammate_cfgs:
            type_name = type_cfg["name"]
            self.type_names.append(type_name)
            self.type_name_to_idx[type_name] = len(self.type_names) - 1
            loader_name = type_cfg["agent_loader"]
            loader_cls = agent_loader_REGISTRY[loader_name]
            path_cfg = type_cfg.get("agent_path", "")
            if path_cfg:
                if os.path.isabs(path_cfg) or os.path.exists(path_cfg):
                    model_path = path_cfg
                else:
                    model_path = os.path.join(base_uncntrl_path, path_cfg)
            else:
                model_path = ""
            expert = loader_cls(
                args=self.args,
                scheme=self.scheme,
                model_path=model_path,
                load_step=type_cfg.get("load_step", "best"),
                load_agent_idx=type_cfg.get("load_agent_idx", 0),
                test_mode=type_cfg.get("test_mode", True),
            )
            self.experts[type_name] = expert

        self.use_param_sharing = True

    # Interface methods -----------------------------------------------------
    def predict(self, ep_batch, agent_idx, t_ep, t_env, bs, test_mode=True):
        env_indices = self._normalize_bs(bs, ep_batch.batch_size)
        if len(env_indices) == 0:
            return None, None, None

        type_predictions = self._predict_types(ep_batch, env_indices, t_ep, bs)
        grouped_envs = defaultdict(list)
        for env in env_indices:
            grouped_envs[type_predictions[env]].append(env)

        combined_logits = None
        combined_actions = None
        combined_hidden = None

        index_lookup = {env: i for i, env in enumerate(env_indices)}

        for type_name, envs in grouped_envs.items():
            expert = self.experts[type_name]
            bs_arg = envs if len(envs) > 1 else [envs[0]]
            logits, actions, hidden = expert.predict(
                ep_batch,
                agent_idx=agent_idx,
                t_ep=t_ep,
                t_env=t_env,
                bs=bs_arg,
                test_mode=test_mode,
            )
            if combined_logits is None:
                shape_logits = (len(env_indices),) + tuple(logits.shape[1:])
                shape_actions = (len(env_indices),) + tuple(actions.shape[1:])
                shape_hidden = (len(env_indices),) + tuple(hidden.shape[1:])
                combined_logits = th.zeros(
                    shape_logits, device=logits.device, dtype=logits.dtype
                )
                combined_actions = th.zeros(
                    shape_actions, device=actions.device, dtype=actions.dtype
                )
                combined_hidden = th.zeros(
                    shape_hidden, device=hidden.device, dtype=hidden.dtype
                )
            for local_idx, env in enumerate(envs):
                pos = index_lookup[env]
                combined_logits[pos].copy_(logits[local_idx])
                combined_actions[pos].copy_(actions[local_idx])
                combined_hidden[pos].copy_(hidden[local_idx])

        return combined_logits, combined_actions, combined_hidden

    def parameters(self):
        params = list(self.classifier.parameters())
        for expert in self.experts.values():
            params += list(expert.parameters())
        return params

    def cuda(self):
        self.classifier.to(self.device)
        for expert in self.experts.values():
            expert.cuda()

    def init_hidden(self, batch_size):
        return None

    # Helper methods --------------------------------------------------------
    def _normalize_bs(self, bs, batch_size):
        if isinstance(bs, slice):
            rng = range(*bs.indices(batch_size))
            return list(rng)
        if isinstance(bs, (list, tuple)):
            return list(bs)
        return [bs]

    def _predict_types(self, ep_batch, env_indices, t_ep, bs):
        batch = ep_batch[bs]
        obs_tensor, time_mask, agent_mask = self._build_classifier_tensors(
            batch, t_ep
        )
        self.classifier.eval()
        with th.no_grad():
            logits = self.classifier(obs_tensor, time_mask, agent_mask)
            preds = logits.argmax(dim=-1)
        mapping = {}
        preds_list = preds.tolist()
        for pos, env in enumerate(env_indices):
            mapping[env] = self.type_names[preds_list[pos]]
        self._update_accuracy(batch, preds_list)
        return mapping

    def _build_classifier_tensors(self, batch, t_ep):
        max_t = batch.max_seq_length if t_ep is None else t_ep + 1
        start = max(0, max_t - self.history_len)
        ts = slice(start, max_t)
        obs = batch["obs"][:, ts].to(self.device)
        time_mask = batch["filled"][:, ts].to(self.device).bool()
        if "trainable_agents" in batch.data.transition_data:
            agent_mask = batch["trainable_agents"][:, ts].to(self.device).bool()
        else:
            mask_shape = obs.shape[:-1] + (1,)
            agent_mask = th.ones(mask_shape, dtype=th.bool, device=self.device)
        return obs, time_mask, agent_mask

    def set_label_mapping(self, name_to_idx: Dict[str, int]):
        cleaned = {}
        for name, idx in name_to_idx.items():
            clean_name = name
            if clean_name.startswith("agent_"):
                clean_name = clean_name[len("agent_"):]
            cleaned[clean_name] = idx
        self.label_mapping = cleaned

    def set_classifier_module(self, module):
        self.classifier = module.to(self.device)

    def get_classifier_module(self):
        return self.classifier

    def _update_accuracy(self, batch, preds_list):
        if not self.label_mapping:
            return
        labels = batch["uncontrolled_team_idx"]
        if labels is None:
            return
        # labels shape: (batch, 1, 1?) depending on scheme
        labels = labels[:, 0]
        if labels.ndim > 1:
            labels = labels[:, 0]
        labels = labels.long()
        pred_labels = []
        for name in [self.type_names[i] for i in preds_list]:
            pred_labels.append(self.label_mapping.get(name, -1))
        pred_tensor = th.tensor(pred_labels, device=labels.device)
        valid = labels >= 0
        if valid.sum() == 0:
            return
        correct = (pred_tensor[valid] == labels[valid]).sum().item()
        total = valid.sum().item()
        self.acc_correct += correct
        self.acc_total += total

    def pop_accuracy(self):
        if self.acc_total == 0:
            return None
        acc = self.acc_correct / self.acc_total
        self.acc_correct = 0
        self.acc_total = 0
        return acc
