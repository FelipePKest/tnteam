import torch as th
import torch.nn.functional as F

from modules.classifiers import build_classifier, classifier_kwargs_from_args
from utils.encoder_decoder import Encoder, get_encoder_input_shape


class ClassifierLearner:
    def __init__(self, mac, scheme, logger, args):
        self.mac = mac
        self.logger = logger
        self.args = args
        obs_dim = scheme["obs"]["vshape"][0] if isinstance(scheme["obs"]["vshape"], tuple) else scheme["obs"]["vshape"]
        self.device = args.device
        self.num_uncontrolled_types = max(1, len(args.uncntrl_agents))
        self.history_len = min(getattr(args, "classifier_history_len", args.episode_limit), args.episode_limit)
        self.classifier_minibatch_size = getattr(args, "classifier_minibatch_size", 1024)
        self.encoder_input_shape = get_encoder_input_shape(scheme)
        self.encoder = Encoder(
            args=self.args,
            input_dim=self.encoder_input_shape,
            hidden_dim=args.ed_hidden_dim,
            output_dim=args.embed_dim,
        ).to(self.device)
        if hasattr(self.mac, "set_encoder"):
            self.mac.set_encoder(self.encoder)
        self.model = build_classifier(
            **classifier_kwargs_from_args(args),
            obs_dim=obs_dim,
            n_agents=args.n_agents,
            episode_limit=args.episode_limit,
            num_uncontrolled_types=self.num_uncontrolled_types,
        ).to(self.device)
        if hasattr(self.mac, "set_classifier"):
            self.mac.set_classifier(self.model)
        self.optimizer = th.optim.Adam(
            self.model.parameters(),
            lr=getattr(args, "classifier_lr", 1e-4),
            weight_decay=getattr(args, "classifier_weight_decay", 0.0),
        )

    def train(self, batch, t_env: int, episode_num: int):
        obs = batch["obs"][:, :-1].to(self.device)
        filled = (batch["filled"][:, :-1].to(self.device) > 0).squeeze(-1).squeeze(-1)

        if "trainable_agents" in batch.data.transition_data:
            agent_mask = batch["trainable_agents"][:, :-1].to(self.device)
            if agent_mask.dim() == 3:
                agent_mask = agent_mask.unsqueeze(-1)
            agent_mask = agent_mask > 0
        else:
            mask_shape = obs.shape[:-1] + (1,)
            agent_mask = th.ones(mask_shape, dtype=th.bool, device=self.device)

        labels = batch["uncontrolled_team_idx"].long().to(self.device).squeeze(-1)

        prepared = self._prepare_training_windows(obs, filled, agent_mask, labels)
        if prepared is None:
            return

        window_obs, window_time_mask, window_agent_mask, window_labels = prepared
        self.model.train()
        num_windows = window_labels.shape[0]
        minibatch_size = max(1, min(self.classifier_minibatch_size, num_windows))
        perm = th.randperm(num_windows, device=self.device)

        self.optimizer.zero_grad()

        total_loss = 0.0
        total_correct = 0
        for start in range(0, num_windows, minibatch_size):
            idx = perm[start:start + minibatch_size]
            logits = self.model(window_obs[idx], window_time_mask[idx], window_agent_mask[idx])
            chunk_labels = window_labels[idx]
            chunk_loss = F.cross_entropy(logits, chunk_labels)
            scaled_loss = chunk_loss * (idx.numel() / num_windows)
            scaled_loss.backward()

            with th.no_grad():
                total_loss += chunk_loss.item() * idx.numel()
                total_correct += (logits.argmax(dim=1) == chunk_labels).sum().item()

        th.nn.utils.clip_grad_norm_(self.model.parameters(), getattr(self.args, "grad_norm_clip", 10))
        self.optimizer.step()

        self.logger.log_stat("classifier_loss", total_loss / num_windows, t_env)
        self.logger.log_stat("classifier_acc", total_correct / num_windows, t_env)

    def cuda(self):
        self.model.to(self.device)
        self.encoder.to(self.device)

    def save_models(self, path):
        th.save(self.model.state_dict(), f"{path}/classifier.th")

    def load_models(self, path):
        state = th.load(f"{path}/classifier.th", map_location=self.device)
        self.model.load_state_dict(state)

    def test(self, batch, t_env: int, log: bool = False):
        """
        Evaluate classifier on test episodes (fresh data, not used for training).
        This provides a true measure of generalization accuracy.
        
        Args:
            batch: Episode batch to evaluate on
            t_env: Current environment timestep (for logging)
            log: Whether to log the accuracy (default False, caller handles logging)
            
        Returns:
            accuracy: Float accuracy on this batch, or None if no valid samples
        """
        self.model.eval()  # Set to eval mode (disables dropout)
        
        obs = batch["obs"][:, :-1].to(self.device)
        filled = (batch["filled"][:, :-1].to(self.device) > 0).squeeze(-1).squeeze(-1)

        if "trainable_agents" in batch.data.transition_data:
            agent_mask = batch["trainable_agents"][:, :-1].to(self.device)
            if agent_mask.dim() == 3:
                agent_mask = agent_mask.unsqueeze(-1)
            agent_mask = agent_mask > 0
        else:
            mask_shape = obs.shape[:-1] + (1,)
            agent_mask = th.ones(mask_shape, dtype=th.bool, device=self.device)

        labels = batch["uncontrolled_team_idx"].long().to(self.device).squeeze(-1)

        prepared = self._prepare_training_windows(obs, filled, agent_mask, labels)
        if prepared is None:
            self.model.train()  # Restore train mode
            return None

        window_obs, window_time_mask, window_agent_mask, window_labels = prepared
        num_windows = window_labels.shape[0]
        minibatch_size = max(1, min(self.classifier_minibatch_size, num_windows))
        
        total_correct = 0
        with th.no_grad():
            for start in range(0, num_windows, minibatch_size):
                end = start + minibatch_size
                logits = self.model(window_obs[start:end], window_time_mask[start:end], window_agent_mask[start:end])
                total_correct += (logits.argmax(dim=1) == window_labels[start:end]).sum().item()
            accuracy = total_correct / num_windows
        
        self.model.train()  # Restore train mode
        
        if log:
            self.logger.log_stat("classifier_test_acc", accuracy, t_env)
        
        return accuracy

    def _prepare_training_windows(self, obs, filled, agent_mask, labels):
        """Construct fixed-horizon windows that mimic evaluation-time queries."""
        batch_size, max_t, n_agents, obs_dim = obs.shape
        history = min(self.history_len, max_t)

        window_obs = []
        window_time_mask = []
        window_agent_mask = []
        window_labels = []

        for b_idx in range(batch_size):
            label = labels[b_idx].item()
            if label < 0:
                continue
            valid_steps = filled[b_idx]
            for t in range(max_t):
                if not valid_steps[t].item():
                    continue
                if not agent_mask[b_idx, t].any():
                    continue

                end = t + 1
                start = max(0, end - history)
                length = end - start

                obs_buf = th.zeros((history, n_agents, obs_dim), device=self.device, dtype=obs.dtype)
                time_buf = th.zeros((history, 1), device=self.device, dtype=th.bool)
                agent_buf = th.zeros((history, n_agents, 1), device=self.device, dtype=th.bool)

                obs_buf[-length:].copy_(obs[b_idx, start:end])
                time_buf[-length:].copy_(filled[b_idx, start:end].unsqueeze(-1))
                agent_buf[-length:].copy_(agent_mask[b_idx, start:end])

                window_obs.append(obs_buf)
                window_time_mask.append(time_buf)
                window_agent_mask.append(agent_buf)
                window_labels.append(label)

        if not window_labels:
            return None

        stacked_obs = th.stack(window_obs, dim=0)
        stacked_time = th.stack(window_time_mask, dim=0)
        stacked_agent = th.stack(window_agent_mask, dim=0)
        stacked_labels = th.tensor(window_labels, device=self.device, dtype=th.long)
        return stacked_obs, stacked_time, stacked_agent, stacked_labels
