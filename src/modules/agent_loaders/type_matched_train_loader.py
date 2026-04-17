"""
TypeMatchedTrainLoader: An agent loader that maintains a pool of pre-trained expert policies
(one per uncontrolled type) and selects the matching policy based on the current uncontrolled type.

This is used for classifier-only training where:
1. The controlled agent policies are FIXED (pre-trained POAM policies)
2. Only the classifier weights are updated
3. The controlled policy matches the uncontrolled type (same as during individual POAM training)

This ensures the training distribution matches the evaluation distribution.
"""
import os
from typing import Dict, List, Optional

import torch as th
from torch import nn

from modules.agent_loaders import REGISTRY as agent_loader_REGISTRY


class TypeMatchedTrainLoader:
    """
    Wrapper that holds a pool of pre-trained POAM policies (one per uncontrolled type).
    The active policy is selected based on the current uncontrolled type index.
    
    Unlike TypeConditionalAgentLoader (for evaluation), this loader:
    - Does NOT use a classifier to predict uncontrolled type
    - Uses ground-truth uncontrolled type (known during training)
    - Keeps all policies frozen (no gradient updates)
    """

    def __init__(
        self,
        args,
        scheme,
        policy_configs: List[Dict],
        base_path: str = "",
    ):
        """
        Args:
            args: Experiment arguments
            scheme: Data scheme
            policy_configs: List of dicts, each with:
                - name: Type name (e.g., "ippo", "qmix")
                - agent_loader: Loader to use (e.g., "poam_eval_agent_loader")
                - agent_path: Path to pre-trained model
                - load_step: Which checkpoint to load (e.g., "best")
                - test_mode: Whether to run in test mode (default True)
            base_path: Base path for resolving relative model paths
        """
        self.args = args
        self.scheme = scheme
        self.device = args.device
        
        # Load all expert policies
        self.type_names: List[str] = []
        self.type_name_to_idx: Dict[str, int] = {}
        self.experts: Dict[str, nn.Module] = {}
        self.experts_by_idx: Dict[int, nn.Module] = {}
        
        for idx, cfg in enumerate(policy_configs):
            type_name = cfg["name"]
            self.type_names.append(type_name)
            self.type_name_to_idx[type_name] = idx
            
            loader_name = cfg["agent_loader"]
            loader_cls = agent_loader_REGISTRY[loader_name]
            
            path_cfg = cfg.get("agent_path", "")
            if path_cfg:
                if os.path.isabs(path_cfg) or os.path.exists(path_cfg):
                    model_path = path_cfg
                else:
                    model_path = os.path.join(base_path, path_cfg)
            else:
                model_path = ""
            
            expert = loader_cls(
                args=self.args,
                scheme=self.scheme,
                model_path=model_path,
                load_step=cfg.get("load_step", "best"),
                load_agent_idx=cfg.get("load_agent_idx", 0),
                test_mode=cfg.get("test_mode", True),
            )
            self.experts[type_name] = expert
            self.experts_by_idx[idx] = expert
        
        # Active policy (set by set_active_type)
        self.active_type_idx: Optional[int] = None
        self.active_expert: Optional[nn.Module] = None
        
        # For compatibility
        self.use_param_sharing = True
        
        # Set first expert as default active
        if len(self.experts_by_idx) > 0:
            self.set_active_type(0)
    
    def set_active_type(self, type_idx: int):
        """Set the active policy based on uncontrolled type index."""
        if type_idx not in self.experts_by_idx:
            raise ValueError(f"Unknown type index {type_idx}, available: {list(self.experts_by_idx.keys())}")
        self.active_type_idx = type_idx
        self.active_expert = self.experts_by_idx[type_idx]
    
    def set_active_type_by_name(self, type_name: str):
        """Set the active policy based on uncontrolled type name."""
        if type_name not in self.type_name_to_idx:
            raise ValueError(f"Unknown type name '{type_name}', available: {list(self.type_name_to_idx.keys())}")
        self.set_active_type(self.type_name_to_idx[type_name])
    
    @property
    def action_selector(self):
        """Return the action selector of the active policy."""
        if self.active_expert is None:
            raise RuntimeError("No active expert set. Call set_active_type first.")
        return self.active_expert.action_selector
    
    @property
    def policy(self):
        """Return the policy of the active expert (for encoder access)."""
        if self.active_expert is None:
            raise RuntimeError("No active expert set. Call set_active_type first.")
        return self.active_expert.policy
    
    def predict(self, ep_batch, agent_idx=None, agent_idx_list=None, t_ep=0, t_env=0, bs=slice(None), test_mode=True):
        """Forward pass using the active expert policy.
        
        Note: POAMEvalAgentLoader only supports single agent_idx, so when agent_idx_list
        is provided, we iterate over each agent and concatenate results.
        """
        if self.active_expert is None:
            raise RuntimeError("No active expert set. Call set_active_type first.")
        
        if agent_idx_list is not None:
            # POAMEvalAgentLoader only supports single agent, so iterate
            all_outs = []
            all_acts = []
            all_hiddens = []
            
            for idx in agent_idx_list:
                out, act, hidden = self.active_expert.predict(
                    ep_batch,
                    agent_idx=idx,
                    t_ep=t_ep,
                    t_env=t_env,
                    bs=bs,
                    test_mode=test_mode
                )
                all_outs.append(out)
                all_acts.append(act)
                all_hiddens.append(hidden)
            
            # Concatenate along agent dimension (dim 2)
            agent_outs = th.cat(all_outs, dim=2)
            chosen_actions = th.cat(all_acts, dim=2)
            hidden_states = th.cat(all_hiddens, dim=2)
            
            return agent_outs, chosen_actions, hidden_states
        else:
            return self.active_expert.predict(
                ep_batch,
                agent_idx=agent_idx,
                t_ep=t_ep,
                t_env=t_env,
                bs=bs,
                test_mode=test_mode
            )
    
    def init_hidden(self, batch_size):
        """Initialize hidden states for the active expert."""
        if self.active_expert is None:
            raise RuntimeError("No active expert set. Call set_active_type first.")
        return self.active_expert.init_hidden(batch_size)
    
    def parameters(self):
        """Return NO parameters - all policies are frozen."""
        # Return empty iterator to indicate no trainable parameters
        return iter([])
    
    def cuda(self):
        """Move all experts to CUDA."""
        for expert in self.experts.values():
            expert.cuda()
    
    def save_models(self, path):
        """No-op - policies are frozen and shouldn't be saved."""
        pass
    
    def load_models(self, path):
        """No-op - policies are loaded at initialization."""
        pass
    
    def state_dict(self):
        """Return empty state dict - policies are frozen."""
        return {}
    
    def load_state_dict(self, state_dict):
        """No-op - policies are frozen."""
        pass
