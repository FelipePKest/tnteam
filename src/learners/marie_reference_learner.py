import numpy as np


class MARIEReferenceLearner:
    """Converts EPyMARL episodes and delegates learning to upstream MARIE."""

    def __init__(self, mac, scheme, logger, args):
        self.mac = mac
        self.logger = logger
        self.args = args
        self.service = getattr(mac, "service", None)
        if self.service is None and hasattr(mac, "trained_agent"):
            self.service = mac.trained_agent.service

    @staticmethod
    def episode_to_rollout(batch, n_agents, batch_index=0):
        # ParallelRunner pads episodes to the longest trajectory in the batch.
        # Derive each rollout's length independently so padding is never stored
        # as experience by the upstream MARIE replay buffer.
        length = int(batch["filled"][batch_index, :-1].sum().item())
        obs = batch["obs"][batch_index, :length].detach().cpu().numpy().astype(np.float32)
        actions = batch["actions_onehot"][batch_index, :length].detach().cpu().numpy().astype(np.float32)
        avail = batch["avail_actions"][batch_index, :length].detach().cpu().numpy().astype(np.float32)
        reward = batch["reward"][batch_index, :length].detach().cpu().numpy().astype(np.float32)
        reward = np.repeat(reward[:, None, :], n_agents, axis=1)
        done = np.zeros((length, n_agents, 1), dtype=np.float32)
        done[-1] = 1.0
        last = np.zeros_like(done)
        last[-1] = 1.0
        return {
            "observation": obs,
            "action": actions,
            "reward": reward,
            "done": done,
            "fake": np.zeros_like(done),
            "avail_action": avail,
            "last": last,
            "controlled": (
                batch["trainable_agents"][batch_index, :length]
                .detach().cpu().numpy().astype(np.float32)
                if "trainable_agents" in batch.scheme
                else np.ones_like(done)
            ),
        }

    def train_episode(self, batch, t_env, episode_num):
        stats = None
        for batch_index in range(batch.batch_size):
            stats = self.service.train(
                self.episode_to_rollout(batch, self.args.n_agents, batch_index)
            )
        if stats is not None:
            self.logger.log_stat("marie_train_count", stats["train_count"], t_env)
            self.logger.log_stat("marie_replay_size", stats["replay_size"], t_env)

    def cuda(self):
        pass

    def save_models(self, path):
        self.mac.save_models(path)

    def load_models(self, path):
        self.mac.load_models(path)

    def close(self):
        self.service.close()
