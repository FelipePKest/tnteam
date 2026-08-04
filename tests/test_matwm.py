from types import SimpleNamespace

import torch as th

from components.episode_buffer import EpisodeBatch, ReplayBuffer
from controllers.matwm_controller import MATWMMAC
from learners.matwm_learner import MATWMLearner


class _Logger:
    def __init__(self):
        self.values = {}

    def log_stat(self, key, value, t):
        self.values[key] = value


def _args():
    return SimpleNamespace(
        n_agents=3,
        n_actions=4,
        hidden_dim=8,
        matwm_hidden_dim=16,
        matwm_n_latents=4,
        matwm_n_categories=4,
        matwm_encoder_hidden_dim=16,
        matwm_encoder_layers=2,
        matwm_transformer_layers=1,
        matwm_teammate_layers=1,
        matwm_attention_heads=4,
        matwm_ff_dim=32,
        matwm_max_seq_length=16,
        matwm_reward_bins=31,
        matwm_agent_hidden_dim=16,
        matwm_context_length=3,
        matwm_agent_batch_size=4,
        matwm_imagination_horizon=2,
        matwm_prefill_steps=3,
        matwm_world_lr=3e-4,
        matwm_agent_lr=3e-4,
        matwm_world_grad_clip=10.0,
        matwm_agent_grad_clip=10.0,
        matwm_dynamics_weight=0.5,
        matwm_representation_weight=0.1,
        matwm_free_bits=0.1,
        matwm_lambda=0.95,
        matwm_entropy_coef=3e-4,
        matwm_ema_decay=0.98,
        gamma=0.99,
        optim_eps=1e-5,
        learner_log_interval=1,
    )


def _batch(args, batch_size=2, length=6):
    scheme = {
        "obs": {"vshape": 5, "group": "agents"},
        "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
        "avail_actions": {
            "vshape": (args.n_actions,), "group": "agents", "dtype": th.int,
        },
        "reward": {"vshape": (1,)},
        "terminated": {"vshape": (1,), "dtype": th.uint8},
        "actor_hidden_states": {"vshape": (args.hidden_dim,), "group": "agents"},
    }
    groups = {"agents": args.n_agents}
    batch = EpisodeBatch(scheme, groups, batch_size, length)
    batch.data.transition_data["obs"].normal_()
    batch.data.transition_data["actions"].random_(args.n_actions)
    batch.data.transition_data["avail_actions"].fill_(1)
    batch.data.transition_data["reward"].normal_()
    batch.data.transition_data["terminated"].zero_()
    batch.data.transition_data["filled"].fill_(1)
    return batch, scheme, groups


def test_matwm_controller_and_training_step_are_finite():
    th.manual_seed(7)
    args = _args()
    batch, scheme, groups = _batch(args)
    mac = MATWMMAC(scheme, groups, args)

    warmup_actions, warmup_hidden = mac.select_actions(batch, 0, 0)
    assert warmup_actions.shape == (2, 1, args.n_agents, 1)
    assert warmup_hidden.shape == (2, 1, args.n_agents, args.hidden_dim)

    policy_actions, _ = mac.select_actions(batch, 2, 10)
    assert policy_actions.shape == warmup_actions.shape

    learner = MATWMLearner(mac, scheme, _Logger(), args)
    before = next(mac.policy.world_model.parameters()).detach().clone()
    learner.train(batch, t_env=10, episode_num=1)
    after = next(mac.policy.world_model.parameters()).detach()
    assert not th.equal(before, after)
    assert all(th.isfinite(parameter).all() for parameter in mac.policy.parameters())


def test_recency_weighted_replay_sampling_returns_valid_batch():
    args = _args()
    episode, scheme, groups = _batch(args, batch_size=1, length=4)
    replay = ReplayBuffer(scheme, groups, buffer_size=5, max_seq_length=4)
    for _ in range(5):
        replay.insert_episode_batch(episode)
    sampled = replay.sample(3, recency_decay=0.9)
    assert sampled.batch_size == 3
    assert sampled.max_seq_length == 4
