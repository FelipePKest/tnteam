"""Transition-capacity model replay and independent episodic policy replay."""
from collections import deque

import numpy as np
import psutil
import torch as th

from components.episode_buffer import EpisodeBatch, ReplayBuffer


class MARIEReplayBuffer:
    """Reference MARIE replay, retaining real records rather than padded slots.

    Model windows can cross terminal boundaries (the learner masks attention).
    Policy history remains episodic and is evicted only at its RAM limit.
    """

    def __init__(self, scheme, groups, capacity, preprocess=None, device='cpu',
                 policy_memory_limit_bytes=30 * 1024**3):
        if capacity < 1:
            raise ValueError('MARIE transition capacity must be positive')
        self.capacity = int(capacity)
        storage = EpisodeBatch(scheme, groups, 1, self.capacity,
                               preprocess=preprocess, device=device)
        self.scheme, self.groups, self.device = storage.scheme, groups, device
        self.records = {k: v[0] for k, v in storage.data.transition_data.items()}
        self.size = self.cursor = 0
        self.visits = {k: np.zeros(self.capacity, dtype=np.int64)
                       for k in ('tokenizer', 'model')}
        self.episodes = deque()
        self.policy_steps = 0
        self.max_policy_steps = None
        self.policy_memory_limit_bytes = policy_memory_limit_bytes
        self._process = psutil.Process()
        self._pools = {}

    def _batch(self, count, length):
        return EpisodeBatch({k: v for k, v in self.scheme.items() if k != 'filled'},
                            self.groups, count, length, device=self.device)

    def insert_episode_batch(self, batch):
        self._pools.clear()
        for b in range(batch.batch_size):
            length = max(0, int(batch['filled'][b].sum()) - 1)
            if length == 0:
                continue
            episode = self._batch(1, length + 1)
            for key, value in episode.data.transition_data.items():
                value.copy_(batch[key][b:b+1, :length+1])
            for key, value in episode.data.episode_data.items():
                value.copy_(batch.data.episode_data[key][b:b+1])
            episode['terminated'][0, length-1] = 1
            if (self.max_policy_steps is None and
                    self._process.memory_info().rss > self.policy_memory_limit_bytes):
                self.max_policy_steps = max(self.policy_steps, length)
            self.episodes.append(episode)
            self.policy_steps += length
            while (self.max_policy_steps is not None and len(self.episodes) > 1
                   and self.policy_steps > self.max_policy_steps):
                self.policy_steps -= self.episodes.popleft().max_seq_length - 1
            # If one episode exceeds capacity, retain its most recent records.
            offset = max(0, length - self.capacity)
            indices = (self.cursor + np.arange(offset, length)) % self.capacity
            tensor_indices = th.as_tensor(indices, device=self.device)
            for key, value in self.records.items():
                value[tensor_indices] = episode[key][0, offset:length]
            for visits in self.visits.values():
                visits[indices] = 0
            self.cursor = (self.cursor + length) % self.capacity
            self.size = min(self.capacity, self.size + length)

    def marie_transition_count(self):
        return self.size

    def _model_pools(self, width, controlled_only):
        key = (width, controlled_only)
        if key not in self._pools:
            order = (np.arange(self.size) + (self.cursor if self.size == self.capacity else 0)) % self.capacity
            if controlled_only:
                if 'trainable_agents' not in self.records:
                    raise ValueError('Controlled replay needs a trainable_agents mask')
                masks = self.records['trainable_agents'][th.as_tensor(order, device=self.device)].squeeze(-1).cpu().numpy()
                # Keep slot identity constant within a cross-episode window.
                # Different team assignments form separate chronological streams.
                # Single-frame tokenizer draws can mix identities, provided
                # the compact agent count is constant across the batch.
                signatures, membership = np.unique(
                    masks.sum(-1, keepdims=True) if width == 1 else masks,
                    axis=0, return_inverse=True)
                pools = [order[membership == i] for i, mask in enumerate(signatures) if mask.any()]
            else:
                pools = [order]
            self._pools[key] = [p for p in pools if len(p) >= width]
        return self._pools[key]

    def sample_marie(self, batch_size, sequence_length, mode='model', temperature='inf',
                     controlled_only=False):
        if mode == 'policy':
            return self._sample_policy(batch_size, sequence_length + 1, controlled_only)
        if mode not in self.visits:
            raise ValueError(f'Unknown MARIE replay mode: {mode}')
        width = max(1, sequence_length)
        pools = [p for p in self._model_pools(width, controlled_only)
                 if len(p) - width + 1 >= batch_size]
        if not pools:
            raise ValueError('Insufficient MARIE transitions for a complete batch')
        sizes = np.array([len(p) - width + 1 for p in pools])
        pool = pools[0] if len(pools) == 1 else pools[np.random.choice(len(pools), p=sizes/sizes.sum())]
        starts = pool[:len(pool)-width+1]
        visits = self.visits[mode][starts].astype(np.float64)
        if temperature in ('inf', float('inf')):
            probabilities = np.ones(len(starts)) if not visits.sum() else 1 - visits/visits.sum()
        else:
            logits = -visits / float(temperature)
            probabilities = np.exp(logits-logits.max())
        probabilities /= probabilities.sum()
        selected = np.random.choice(len(starts), batch_size, replace=False, p=probabilities)
        indices = pool[selected[:, None] + np.arange(width)]
        # The existing learner API expects L+1 frames. Reference loss uses the
        # first L transition records only; the last frame is a shape sentinel.
        if sequence_length:
            indices = np.concatenate((indices, indices[:, -1:]), axis=1)
        result = self._batch(batch_size, indices.shape[1])
        indices = th.as_tensor(indices, device=self.device)
        for key, value in self.records.items():
            result.data.transition_data[key].copy_(value[indices])
        result.marie_reference_records = True
        self.visits[mode][starts[selected]] += 1
        if controlled_only:
            from components.marie_context import compact_controlled
            result = compact_controlled(result)
            result.marie_reference_records = True
        return result

    def _sample_policy(self, count, length, controlled_only):
        episodes = list(self.episodes)
        if not episodes:
            raise ValueError('Cannot sample empty MARIE policy replay')
        if controlled_only:
            counts = np.array([int(e['trainable_agents'][0, 0].sum()) for e in episodes])
            keep = ReplayBuffer._controlled_pool(counts, 1)
            episodes = [e for e, selected in zip(episodes, keep) if selected]
        result = self._batch(count, length)
        for row, idx in enumerate(np.random.choice(len(episodes), count, replace=True)):
            episode = episodes[idx]
            stop = np.random.randint(1, episode.max_seq_length)
            start = max(0, stop-length)
            target = slice(length-(stop-start), length)
            for key, value in result.data.transition_data.items():
                value[row, target] = episode[key][0, start:stop]
            for key, value in result.data.episode_data.items():
                value[row] = episode.data.episode_data[key][0]
        if controlled_only:
            from components.marie_context import compact_controlled
            result = compact_controlled(result)
        return result
