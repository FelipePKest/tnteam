"""Shared helpers for the teammate-type probe (collect + fit).

Run from the repo root with PYTHONPATH=src, e.g.
    PYTHONPATH=src python scripts/probe_fit.py --help
"""

import json
import os
from types import SimpleNamespace

import torch as th
import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as stream:
        return yaml.load(stream, Loader=yaml.FullLoader) or {}


def sacred_config_for_checkpoint(checkpoint_dir):
    """Locate <run>/sacred/<run>/1/config.json for a models/<run>/<step> folder."""
    run_dir = os.path.dirname(checkpoint_dir.rstrip("/"))  # models/<run>
    candidates = [
        os.path.join(run_dir.replace("models", "sacred"), "1", "config.json"),
        os.path.join(run_dir, "config.json"),
        os.path.join(checkpoint_dir, "config.json"),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return None


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as stream:
        return json.load(stream)


def default_marie_config(map_name="3s_vs_5z"):
    """Fallback config built from the repo yamls (used for synthetic tests)."""
    config = {}
    for relative in (
        "src/config/default/default.yaml",
        "src/config/open/open_train_3sv5z.yaml",
        "src/config/envs/sc2.yaml",
        "src/config/algs/sc2/marie.yaml",
    ):
        path = os.path.join(REPO, relative)
        if os.path.exists(path):
            config.update(load_yaml(path))
    config.setdefault("env_args", {})
    config["env_args"]["map_name"] = map_name
    config["seed"] = 0
    return config


def build_policy(config, obs_dim, n_agents, n_actions, state_dict_path=None):
    from modules.agents.marie_policy import MARIEPolicy

    args = SimpleNamespace(**config)
    args.n_agents = n_agents
    args.n_actions = n_actions
    args.device = "cpu"
    args.use_cuda = False
    policy = MARIEPolicy(obs_dim, n_agents, n_actions, args)
    if state_dict_path is not None:
        state = th.load(state_dict_path, map_location="cpu")
        missing, unexpected = policy.load_state_dict(state, strict=False)
        if missing or unexpected:
            print("[probe] load_state_dict: missing=%d unexpected=%d" % (len(missing), len(unexpected)))
            for key in list(missing)[:10]:
                print("   missing:", key)
            for key in list(unexpected)[:10]:
                print("   unexpected:", key)
    policy.eval()
    return policy


@th.no_grad()
def world_model_hidden(policy, obs_frames, actions, n_controlled):
    """Aggregation-token state after the last complete (obs, action) block.

    obs_frames: [K, L, obs_dim] frames 0..L-1 of the K controlled agents
    actions:    [K, L, 1] actions taken at those frames
    Mirrors ``controlled_group``: the Perceiver aggregates the K controlled
    streams relabelled 0..K-1, exactly as in controlled-only training.
    Returns [K, hidden_dim].
    """
    world = policy.world_model
    window = world.max_seq_length
    obs_frames = obs_frames[:, -window:]
    actions = actions[:, -window:]
    previous = getattr(world, "context_n_agents", None)
    world.context_n_agents = n_controlled
    try:
        latent, _ = world.encode(obs_frames, sample=False)
        focal = th.arange(n_controlled)
        hidden = world.dynamics_sequence(latent, actions, focal)[:, -1]
    finally:
        if previous is None:
            del world.context_n_agents
        else:
            world.context_n_agents = previous
    return hidden


@th.no_grad()
def reconstructed_stack(policy, obs_frames):
    """What the MARIE actor sees: tokenizer reconstruction, clamped, 5-frame stack."""
    recon = policy.reconstruct_observations(obs_frames)
    return policy.build_state_from_reconstructed(recon)


def raw_stack(obs_frames, n_frames):
    """Zero-left-padded stack of the last ``n_frames`` raw observations."""
    count, length, dim = obs_frames.shape
    frames = obs_frames[:, -n_frames:]
    if frames.shape[1] < n_frames:
        pad = frames.new_zeros(count, n_frames - frames.shape[1], dim)
        frames = th.cat((pad, frames), dim=1)
    return frames.reshape(count, -1)
