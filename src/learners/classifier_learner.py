import torch as th
import torch.nn.functional as F

from modules.classifiers import UncontrolledTransformerClassifier
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
        self.encoder_input_shape = get_encoder_input_shape(scheme)
        self.encoder = Encoder(
            args=self.args,
            input_dim=self.encoder_input_shape,
            hidden_dim=args.ed_hidden_dim,
            output_dim=args.embed_dim,
        ).to(self.device)
        if hasattr(self.mac, "set_encoder"):
            self.mac.set_encoder(self.encoder)
        self.model = UncontrolledTransformerClassifier(
            obs_dim=obs_dim,
            n_agents=args.n_agents,
            episode_limit=args.episode_limit,
            num_uncontrolled_types=self.num_uncontrolled_types,
            d_model=getattr(args, "classifier_d_model", 128),
            nhead=getattr(args, "classifier_nhead", 4),
            num_layers=getattr(args, "classifier_layers", 2),
            dim_feedforward=getattr(args, "classifier_ff", 256),
            dropout=getattr(args, "classifier_dropout", 0.1),
        ).to(self.device)
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
        logits = self.model(window_obs, window_time_mask, window_agent_mask)
        loss = F.cross_entropy(logits, window_labels)

        self.optimizer.zero_grad()
        loss.backward()
        th.nn.utils.clip_grad_norm_(self.model.parameters(), getattr(self.args, "grad_norm_clip", 10))
        self.optimizer.step()

        with th.no_grad():
            preds = logits.argmax(dim=1)
            accuracy = (preds == window_labels).float().mean()
        self.logger.log_stat("classifier_loss", loss.item(), t_env)
        self.logger.log_stat("classifier_acc", accuracy.item(), t_env)

    def cuda(self):
        self.model.to(self.device)
        self.encoder.to(self.device)

    def save_models(self, path):
        th.save(self.model.state_dict(), f"{path}/classifier.th")

    def load_models(self, path):
        state = th.load(f"{path}/classifier.th", map_location=self.device)
        self.model.load_state_dict(state)

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
