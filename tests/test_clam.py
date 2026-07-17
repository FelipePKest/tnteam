from types import SimpleNamespace

import torch as th

from components.episode_buffer import EpisodeBatch
from components.transforms import OneHot
from controllers.agent_owned_controller import AgentOwnedMAC
from learners.clam_learner import CLAMLearner
from learners.clam_type_supervised_learner import TypeSupervisedCLAMLearner
from modules.agents.rnn_clam_agent import RNNCLAMAgent
from modules.clam import (
    CLAMEncoder,
    CLAMProjectionHead,
    supervised_contrastive_loss,
)
from modules.critics.clam import CLAMCritic


def make_args(**overrides):
    values = {
        "n_agents": 3,
        "n_actions": 5,
        "episode_limit": 6,
        "embed_dim": 8,
        "hidden_dim": 16,
        "clam_model_dim": 16,
        "clam_n_heads": 4,
        "clam_n_layers": 2,
        "clam_ff_dim": 32,
        "clam_dropout": 0.0,
        "clam_projection_hidden_dim": 16,
        "clam_projection_dim": 6,
        "use_rnn": True,
        "use_obs_norm": False,
        "use_orthogonal_init": True,
        "obs_last_action": True,
        "obs_agent_id": True,
        "obs_state": False,
        "use_popart": False,
        "batch_size": 2,
        "device": "cpu",
        "env": "sc2",
        "env_args": {},
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class FakeBatch:
    def __init__(self, data):
        self._data = data
        self.batch_size = data["obs"].size(0)
        self.max_seq_length = data["obs"].size(1)
        self.device = data["obs"].device
        self.data = SimpleNamespace(transition_data=data)

    def __getitem__(self, key):
        return self._data[key]


def make_batch(args, obs_dim=7, max_t=5):
    batch_size = args.batch_size
    observations = th.randn(batch_size, max_t, args.n_agents, obs_dim)
    actions = th.randint(
        args.n_actions, (batch_size, max_t, args.n_agents, 1)
    )
    actions_onehot = th.zeros(
        batch_size, max_t, args.n_agents, args.n_actions
    )
    actions_onehot.scatter_(-1, actions, 1)
    return FakeBatch(
        {
            "obs": observations,
            "actions_onehot": actions_onehot,
            "actor_hidden_states": th.zeros(
                batch_size, max_t, args.n_agents, args.hidden_dim
            ),
            "filled": th.ones(batch_size, max_t, 1),
        }
    )


def test_prefix_encoding_matches_online_encoding():
    th.manual_seed(7)
    args = make_args()
    encoder = CLAMEncoder(obs_dim=7, args=args).eval()
    observations = th.randn(2, 5, args.n_agents, 7)

    all_prefixes = encoder.forward_prefixes(observations)
    for timestep in range(observations.size(1)):
        prefix = observations[:, : timestep + 1].permute(0, 2, 1, 3)
        prefix = prefix.reshape(-1, timestep + 1, 7)
        online = encoder(prefix).view(2, args.n_agents, args.embed_dim)
        th.testing.assert_close(all_prefixes[:, timestep], online)


def test_contrastive_projection_backpropagates():
    th.manual_seed(11)
    args = make_args()
    encoder = CLAMEncoder(obs_dim=7, args=args)
    projector = CLAMProjectionHead(args.embed_dim, args)
    first = projector(encoder(th.randn(4, 5, 7)))
    second = projector(encoder(th.randn(4, 4, 7)))

    labels = th.tensor([0, 0, 1, 1])
    loss = supervised_contrastive_loss(
        first, second, labels=labels, temperature=0.5
    )
    loss.backward()

    assert th.isfinite(loss)
    assert any(parameter.grad is not None for parameter in encoder.parameters())


def test_type_supervision_uses_cross_episode_positives():
    labels = th.tensor([0, 0, 1, 1])
    first = th.tensor(
        [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]
    )
    aligned_second = first.clone()
    misaligned_second = th.flip(first, dims=[0])

    aligned_loss = supervised_contrastive_loss(
        first, aligned_second, labels, temperature=0.1
    )
    misaligned_loss = supervised_contrastive_loss(
        first, misaligned_second, labels, temperature=0.1
    )

    assert aligned_loss < misaligned_loss


def test_actor_and_critic_are_conditioned_on_clam_context():
    th.manual_seed(13)
    args = make_args()
    obs_dim = 7
    input_shape = obs_dim + args.n_actions + args.n_agents
    batch = make_batch(args, obs_dim=obs_dim)
    encoder = CLAMEncoder(obs_dim=obs_dim, args=args).eval()

    actor = RNNCLAMAgent(input_shape, args)
    actor.encoder = encoder
    logits, actor_hidden = actor(batch, t=None)
    assert logits.shape == (2, 5, args.n_agents, args.n_actions)
    assert actor_hidden.shape == (2, 5, args.n_agents, args.hidden_dim)

    scheme = {
        "obs": {"vshape": obs_dim},
        "state": {"vshape": 9},
        "actions_onehot": {"vshape": (args.n_actions,)},
    }
    critic = CLAMCritic(scheme, args)
    critic.encoder = encoder
    values, critic_hidden = critic(
        batch,
        th.zeros(2, 5, args.n_agents, args.hidden_dim),
        t=None,
    )
    assert values.shape == (2, 5, args.n_agents, 1)
    assert critic_hidden.shape == (2, 5, args.n_agents, args.hidden_dim)


class FakeLogger:
    def __init__(self):
        self.stats = {}

    def log_stat(self, key, value, timestep):
        self.stats[key] = (value, timestep)


def test_type_supervised_clam_learner_runs_one_joint_update(tmp_path):
    args = make_args(
        batch_size=4,
        open_train_or_eval=True,
        trainable_agents_mask_actor=True,
        trainable_agents_mask_critic=False,
        action_selector="soft_policies",
        agent_output_type="pi_logits",
        mask_before_softmax=True,
        agent="rnn_clam",
        critic_type="clam_critic",
        lr=5e-4,
        optim_eps=1e-5,
        standardise_rewards=False,
        mask_type="team",
        use_adv_std=True,
        use_gae=True,
        gamma=0.99,
        gae_lambda=0.95,
        q_nstep=5,
        add_value_last_step=True,
        epochs=1,
        n_minibatch=1,
        eps_clip=0.1,
        entropy_coef=0.01,
        grad_norm_clip=10,
        use_huber_loss=False,
        huber_delta=10.0,
        clip_value_loss=False,
        clam_lr=0.003,
        clam_temperature=0.5,
        clam_instance_temperature=0.2,
        clam_supervised_temperature=0.2,
        clam_instance_coef=1.0,
        clam_supervised_coef=1.0,
        clam_mask_ratio=0.3,
        clam_min_crop=2,
        clam_max_crop=4,
        clam_prefix_crop_probability=1.0,
        clam_target_tau=0.01,
        clam_update_interval=1,
        clam_grad_norm_clip=5,
        learner_log_interval=1,
        use_cuda=False,
    )
    obs_dim = 7
    state_dim = 9
    max_t = 5
    scheme = {
        "state": {"vshape": state_dim},
        "obs": {"vshape": obs_dim, "group": "agents"},
        "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
        "actor_hidden_states": {
            "vshape": (args.hidden_dim,),
            "group": "agents",
        },
        "avail_actions": {
            "vshape": (args.n_actions,),
            "group": "agents",
            "dtype": th.int,
        },
        "reward": {"vshape": (1,)},
        "terminated": {"vshape": (1,), "dtype": th.uint8},
        "trainable_agents": {
            "vshape": (1,),
            "group": "agents",
            "dtype": th.bool,
        },
        "uncontrolled_team_idx": {
            "vshape": (1,),
            "dtype": th.long,
            "episode_const": True,
        },
    }
    groups = {"agents": args.n_agents}
    preprocess = {"actions": ("actions_onehot", [OneHot(args.n_actions)])}
    batch = EpisodeBatch(
        scheme,
        groups,
        args.batch_size,
        max_t,
        preprocess=preprocess,
    )
    batch.update(
        {
            "uncontrolled_team_idx": th.tensor(
                [[0], [0], [1], [1]], dtype=th.long
            )
        }
    )
    for timestep in range(max_t):
        batch.update(
            {
                "state": th.randn(args.batch_size, state_dim),
                "obs": th.randn(args.batch_size, args.n_agents, obs_dim),
                "avail_actions": th.ones(
                    args.batch_size, args.n_agents, args.n_actions, dtype=th.int
                ),
                "actor_hidden_states": th.zeros(
                    args.batch_size, args.n_agents, args.hidden_dim
                ),
                "trainable_agents": th.ones(
                    args.batch_size, args.n_agents, 1, dtype=th.bool
                ),
            },
            ts=timestep,
        )
        if timestep < max_t - 1:
            batch.update(
                {
                    "actions": th.randint(
                        args.n_actions,
                        (args.batch_size, args.n_agents, 1),
                    ),
                    "reward": th.randn(args.batch_size, 1),
                    "terminated": th.zeros(args.batch_size, 1, dtype=th.uint8),
                },
                ts=timestep,
                mark_filled=False,
            )

    mac = AgentOwnedMAC(batch.scheme, groups, args)
    logger = FakeLogger()
    learner = TypeSupervisedCLAMLearner(mac, batch.scheme, logger, args)
    learner.train(batch, t_env=10, episode_num=0)

    assert "clam_loss" in logger.stats
    assert "clam_instance_loss" in logger.stats
    assert "clam_supervised_loss" in logger.stats
    combined_loss = (
        args.clam_instance_coef * logger.stats["clam_instance_loss"][0]
        + args.clam_supervised_coef * logger.stats["clam_supervised_loss"][0]
    )
    assert abs(logger.stats["clam_loss"][0] - combined_loss) < 1e-5
    assert "clam_type_cosine_margin" in logger.stats
    assert logger.stats["clam_prefix_views"][0] == 1.0
    assert logger.stats["clam_num_types"][0] == 2.0
    assert "actor_loss" in logger.stats
    assert "critic_loss" in logger.stats

    learner.save_models(tmp_path)
    restored_mac = AgentOwnedMAC(batch.scheme, groups, args)
    restored = TypeSupervisedCLAMLearner(
        restored_mac, batch.scheme, FakeLogger(), args
    )
    restored.load_models(tmp_path)
    for expected, actual in zip(
        learner.clam_encoder.parameters(), restored.clam_encoder.parameters()
    ):
        th.testing.assert_close(expected, actual)

    # The original learner remains independent of uncontrolled-team labels
    # and retains its episode-level InfoNCE objective.
    original_args = SimpleNamespace(**vars(args))
    original_args.open_train_or_eval = False
    original_mac = AgentOwnedMAC(batch.scheme, groups, original_args)
    original = CLAMLearner(
        original_mac, batch.scheme, FakeLogger(), original_args
    )
    original_stats = original._contrastive_update(batch)
    assert "clam_positive_cosine" in original_stats
    assert "clam_type_cosine_margin" not in original_stats
