"""Compact controlled-agent streams while keeping each sampled team together."""
from contextlib import contextmanager
from functools import wraps

import torch as th
from components.episode_buffer import EpisodeBatch


def compact_controlled(batch):
    mask = batch.data.transition_data.get('trainable_agents')
    if mask is None:
        raise ValueError('Controlled-only replay requires trainable_agents')
    # Policy contexts can be left padded; their final frame is always valid.
    selected = mask[:, -1, :, 0].bool()
    counts = selected.sum(-1)
    if not bool((counts > 0).all()) or not bool((counts == counts[0]).all()):
        raise ValueError('Controlled contexts must have the same positive agent count per batch')
    count = int(counts[0])
    indices = selected.nonzero()[:, 1].reshape(batch.batch_size, count)
    # Drop global state, foreign policy identities and all unrelated metadata.
    fields = {'obs', 'actions', 'actions_onehot', 'avail_actions',
              'trainable_agents', 'reward', 'terminated'}
    scheme = {k: v for k, v in batch.scheme.items() if k in fields}
    groups = dict(batch.groups, agents=count)
    result = EpisodeBatch(scheme, groups, batch.batch_size, batch.max_seq_length,
                          preprocess=None, device=batch.device)
    for key, target in result.data.transition_data.items():
        source = batch[key]
        if key != 'filled' and batch.scheme[key].get('group') == 'agents':
            index = indices[:, None].reshape(batch.batch_size, 1, count, *([1]*(source.ndim-3)))
            shape = list(source.shape)
            shape[2] = count
            source = source.gather(2, index.expand(shape))
        target.copy_(source)
    return result


@contextmanager
def controlled_group(learner, count):
    """Use compact group sizes only inside training/validation, never execution.

    Slots are relabelled 0..K-1 within the controlled subteam. Real environment
    agent indices and the shared model parameter shapes remain unchanged.
    """
    previous = (learner.n_agents, learner.policy.n_agents,
                getattr(learner.world_model, 'context_n_agents', None))
    learner.n_agents = learner.policy.n_agents = count
    learner.world_model.context_n_agents = count
    try:
        yield
    finally:
        learner.n_agents, learner.policy.n_agents = previous[:2]
        if previous[2] is None:
            del learner.world_model.context_n_agents
        else:
            learner.world_model.context_n_agents = previous[2]


def controlled_training(method):
    @wraps(method)
    def wrapped(self, batch, *args, **kwargs):
        if not getattr(self.args, 'marie_controlled_replay_only', False):
            return method(self, batch, *args, **kwargs)
        mask = batch.data.transition_data.get('trainable_agents')
        if mask is None or not bool(mask[:, -1].all()):
            raise ValueError('Controlled-only learner received an unfiltered replay context')
        with controlled_group(self, batch['obs'].shape[2]):
            return method(self, batch, *args, **kwargs)
    return wrapped
