from modules.agent_loaders import REGISTRY as agent_loader_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
import os
import random
import numpy as np
import torch as th

class OpenTrainMAC:
    def __init__(self, scheme, groups, args):
        '''This class was based off the OpenEvalMAC'''
        self.n_agents = args.n_agents
        self.args = args
        self.n_uncontrolled = args.n_uncontrolled
        self._build_agent_pool(scheme)
        self.active_uncontrolled_team = None
        self.active_uncontrolled_team_idx = None
        self.sample_agent_team()

        # hacky way to provide compatibility with learners
        self.action_selector = self.trained_agent.action_selector

    def select_actions(self, ep_batch,
                       t_ep, t_env, bs=slice(None), 
                       test_mode=False):
        '''Select joint action using the active team'''
        trained_agent_idxs = [agent_idx for agent_idx, _, team_name in self._active_team if team_name == "trained_agent_subteam"]
        _, trained_agent_act, trained_agent_hidden = self._predict_trained_agents(
            ep_batch,
            trained_agent_idxs,
            t_ep=t_ep,
            t_env=t_env,
            bs=bs,
            test_mode=test_mode,
        )
        
        # compile outputs
        curr_agent_idx = 0
        joint_act = []
        joint_hidden = []
        for agent_idx, subteam_idx, team_name in self._active_team:
            if team_name == "uncontrolled_agent_subteam":
                agent = self.uncontrolled_agent_pool[subteam_idx]
                # uncontrolled agents should be evaluated in test mode
                _, act, hidden_state = agent.predict(ep_batch, agent_idx=agent_idx, 
                                                     t_ep=t_ep, t_env=t_env, bs=bs
                                                     )
            else:
                assert team_name == "trained_agent_subteam"
                act = trained_agent_act[:, :, slice(curr_agent_idx, curr_agent_idx+1)]
                hidden_state = trained_agent_hidden[:, :, slice(curr_agent_idx, curr_agent_idx+1)]
                curr_agent_idx += 1
            
            joint_act.append(act)
            joint_hidden.append(hidden_state)
            
        joint_act = th.cat(joint_act, dim=2)
        joint_hidden = th.cat(joint_hidden, dim=2)
        return joint_act, joint_hidden
    
    def forward(self, ep_batch, t=None, test_mode=False):
        '''This function is used by learners only. Thus, we only execute the forward pass 
        using the trained agent.'''
        trained_agent_idxs = list(range(self.n_agents))
        # t_env is irrelevant here because only logits are consumed by the learner.
        agent_out, _, hidden = self._predict_trained_agents(
            ep_batch,
            trained_agent_idxs,
            t_ep=t,
            t_env=0,
            bs=slice(None),
            test_mode=test_mode,
        )
        
        return agent_out, hidden

    def _predict_trained_agents(self, ep_batch, trained_agent_idxs, t_ep, t_env, bs, test_mode):
        if not getattr(self, "_use_type_conditional_loader", False):
            return self.trained_agent.predict(
                ep_batch,
                agent_idx_list=trained_agent_idxs,
                t_ep=t_ep,
                t_env=t_env,
                bs=bs,
                test_mode=test_mode,
            )

        trained_outputs = [
            self.trained_agent.predict(
                ep_batch,
                agent_idx=agent_idx,
                t_ep=t_ep,
                t_env=t_env,
                bs=bs,
                test_mode=test_mode,
            )
            for agent_idx in trained_agent_idxs
        ]
        agent_out = th.cat([out[0] for out in trained_outputs], dim=2)
        actions = th.cat([out[1] for out in trained_outputs], dim=2)
        hidden = th.cat([out[2] for out in trained_outputs], dim=2)
        return agent_out, actions, hidden

    def set_encoder(self, encoder): 
        if hasattr(self.trained_agent, "set_encoder"):
            self.trained_agent.set_encoder(encoder)
        else:
            self.trained_agent.policy.encoder = encoder

    def set_classifier(self, classifier):
        if hasattr(self.trained_agent, "set_classifier_module"):
            self.trained_agent.set_classifier_module(classifier)
        
    def init_hidden(self, batch_size):
        '''A dummy function for open evaluation only.'''
        return th.zeros(batch_size, 1, self.n_agents, self.args.hidden_dim)

    def parameters(self):
        '''Return learnable parameters'''
        return self.trained_agent.parameters()
    
    def load_state(self, other_mac):
        '''Used by the Q-learning, QMIX, QTRAN and MADDPG learners'''
        self.trained_agent.load_state_dict(other_mac.trained_agent.state_dict())

    def cuda(self):
        for agent in [self.trained_agent, *self.uncontrolled_agent_pool]:
            agent.cuda()
    
    def save_models(self, path):
        self.trained_agent.save_models(path)
    
    def load_models(self, path):
        # TODO check if this is correct
        self.trained_agent.load_state_dict(th.load("{}/agent.th".format(path)))

    def sample_agent_team(self): 
        '''
        This function controls the openness of the evaluation.
        Randomly samples n_uncontrolled agents from the uncontrolled agent team.
        ''' 
        # sample number of uncontrolled agents
        if self.n_uncontrolled is None: # sample n_uncontrolled uniformly from 1 to n_agents-1
            n_uncontrolled = np.random.randint(1, self.n_agents)
        else:
            n_uncontrolled = self.n_uncontrolled
        # sample uncontrolled agent team
        active_uncontrolled_team = np.random.choice(list(self.uncontrolled_agent_teams.keys()))
        self.active_uncontrolled_team = active_uncontrolled_team
        if self.uncontrolled_team_name_to_idx is not None:
            self.active_uncontrolled_team_idx = self.uncontrolled_team_name_to_idx[active_uncontrolled_team]
        else:
            self.active_uncontrolled_team_idx = None
        
        # Type-matched training hard-switches from the ground-truth label.
        # Type-conditional training keeps the eval-time feedback loop and
        # lets the classifier prediction choose the active expert instead.
        if getattr(self, '_use_type_matched_loader', False) and self.active_uncontrolled_team_idx is not None:
            self.trained_agent.set_active_type(self.active_uncontrolled_team_idx)
        
        self.uncontrolled_agent_pool = self.uncontrolled_agent_teams[active_uncontrolled_team]
        uncontrolled_agent_idxs = list(np.random.choice(len(self.uncontrolled_agent_pool), 
                                                     n_uncontrolled, 
                                                     replace=False))
        trained_agent_idxs = list(np.random.choice(range(self.n_agents), 
                                                   self.n_agents - n_uncontrolled, 
                                                   replace=False))
        # order agents from uncontrolled and trained teams randomly
        agent_order = list(range(self.n_agents))
        random.shuffle(agent_order)
        self._active_team = [(agent_order.pop(0), i, "uncontrolled_agent_subteam") for i in uncontrolled_agent_idxs] + \
                            [(agent_order.pop(0), i, "trained_agent_subteam") for i in trained_agent_idxs]
        
        # original agent order
        # self._active_team = [(i, i, "trained_agent_subteam") for i in range(self.n_agents)]
        # shuffled agent order
        # self._active_team = [(agent_order.pop(0), i, "trained_agent_subteam") for i in range(self.n_agents)]

        self._active_team = sorted(self._active_team, key=lambda x: x[0])

        # indices of the trained agents
        trained_agent_idxs = [agent_idx for agent_idx, _, team_name in self._active_team if team_name == "trained_agent_subteam"]
        return trained_agent_idxs

    def _build_agent_pool(self, scheme):
        '''
        Example yaml to be loaded into args: 
        base_checkpoint_path: ""
        trained_agents:
            agent_0:
                agent_loader: "rnn_train_agent_loader"
                agent_path: "" # leave empty for training from scratch
        uncntrl_agents:
            agent_0:
                agent_loader: "rnn_eval_agent_loader"
                agent_path: ""
                n_agents_to_populate: 5
                load_step: best
        '''
        # initialize training agents
        agent_loader = self.args.trained_agents['agent_0']['agent_loader']
        
        self.classifier_agents = []

        if agent_loader == "type_matched_train_loader":
            policy_configs = self.args.trained_agents['agent_0'].get('policy_configs', [])
            base_path = self.args.trained_agents['agent_0'].get('base_path', '')
            self.trained_agent = agent_loader_REGISTRY[agent_loader](
                args=self.args,
                scheme=scheme,
                policy_configs=policy_configs,
                base_path=base_path,
            )
            self._use_type_matched_loader = True
            self._use_type_conditional_loader = False
        elif agent_loader == "type_conditional_loader":
            classifier_cfg = self.args.trained_agents['agent_0'].get("classifier", {})
            teammate_cfgs = self.args.trained_agents['agent_0'].get("teammate_types", [])
            base_path = self.args.trained_agents['agent_0'].get("base_path", self.args.base_uncntrl_path)
            self.trained_agent = agent_loader_REGISTRY[agent_loader](
                args=self.args,
                scheme=scheme,
                classifier_cfg=classifier_cfg,
                teammate_cfgs=teammate_cfgs,
                base_uncntrl_path=base_path,
            )
            self._use_type_matched_loader = False
            self._use_type_conditional_loader = True
            self.classifier_agents = [self.trained_agent]
        else:
            agent_path = self.args.trained_agents['agent_0']['agent_path']
            self.trained_agent = agent_loader_REGISTRY[agent_loader](
                args=self.args,
                scheme=scheme,
                model_path=agent_path
            )
            self._use_type_matched_loader = False
            self._use_type_conditional_loader = False

        # initialize+load uncontrolled agents
        base_uncntrl_path = self.args.base_uncntrl_path
        uncntrl_agents_dict = self.args.uncntrl_agents
        self.uncontrolled_agent_teams = {}
        self.uncontrolled_team_name_to_idx = None

        for agent_nm, agent_cfg in uncntrl_agents_dict.items():
            self.uncontrolled_agent_teams[agent_nm] = []
            use_param_sharing = False

            loader_key = agent_cfg["agent_loader"]
            if loader_key == "type_conditional_loader":
                classifier_cfg = agent_cfg.get("classifier", {})
                teammate_cfgs = agent_cfg.get("teammate_types", [])
                agent = agent_loader_REGISTRY[loader_key](
                    args=self.args,
                    scheme=scheme,
                    classifier_cfg=classifier_cfg,
                    teammate_cfgs=teammate_cfgs,
                    base_uncntrl_path=base_uncntrl_path,
                )
                copies = agent_cfg.get("n_agents_to_populate", self.n_agents - 1)
                copies = max(1, copies)
                for _ in range(copies):
                    self.uncontrolled_agent_teams[agent_nm].append(agent)
                continue

            n_populate = agent_cfg.get("n_agents_to_populate", self.n_agents - 1)
            assert n_populate >= self.n_agents - 1
            for i in range(n_populate):
                # load in new agent only if param sharing is not used
                # else, python will place a reference to the single agent in all team slots
                if not use_param_sharing: 
                    if agent_cfg["agent_loader"] == "bot_agent_loader":
                        agent = agent_loader_REGISTRY[agent_cfg['agent_loader']](
                            args=self.args, scheme=scheme,
                            bot_name=agent_cfg["bot_name"],
                        )
                    else:
                        model_path = os.path.join(base_uncntrl_path, agent_cfg["agent_path"])
                        agent = agent_loader_REGISTRY[agent_cfg['agent_loader']](args=self.args, 
                                                                                scheme=scheme, 
                                                                                model_path=model_path, 
                                                                                load_step=agent_cfg["load_step"],
                                                                                load_agent_idx=i, # only matters for ns methods
                                                                                test_mode=agent_cfg["test_mode"]
                                                                                )
                        use_param_sharing = agent.use_param_sharing
                self.uncontrolled_agent_teams[agent_nm].append(agent)

        if len(self.uncontrolled_agent_teams) > 0:
            self.uncontrolled_team_name_to_idx = {
                name: idx for idx, name in enumerate(self.uncontrolled_agent_teams.keys())
            }

        for agent in getattr(self, "classifier_agents", []):
            if hasattr(agent, "set_label_mapping") and self.uncontrolled_team_name_to_idx is not None:
                agent.set_label_mapping(self.uncontrolled_team_name_to_idx)

    def get_classifier_accuracy(self):
        accuracies = []
        for agent in getattr(self, "classifier_agents", []):
            if hasattr(agent, "pop_accuracy"):
                acc = agent.pop_accuracy()
                if acc is not None:
                    accuracies.append(acc)
        if not accuracies:
            return None
        return sum(accuracies) / len(accuracies)
