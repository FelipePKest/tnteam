#!/usr/bin/env python
"""Collect labelled open-team episodes with a frozen MARIE/LION checkpoint.

For every teammate type in the run's config, and for each requested number of
uncontrolled agents K, roll out N test episodes and store the controlled agents'
observations/actions together with the teammate-type label. Output is a single
torch file consumed by scripts/probe_fit.py.

Example (from repo root):
    PYTHONPATH=src python scripts/probe_collect.py \
        --checkpoint naht_results/3sv5z/open_train/<run>/models/<expt>/best \
        --episodes-per-type 40 --n-uncontrolled 1 2 \
        --out probe_data/3sv5z_seed123.pt

The sacred config.json of the run is located automatically next to the
checkpoint (models -> sacred). Pass --config to override.
"""

import argparse
import copy
import os
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch as th

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from probe_common import REPO, load_config, load_yaml, sacred_config_for_checkpoint  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, help="folder containing agent.th (e.g. .../models/<expt>/best)")
    parser.add_argument("--config", default=None, help="sacred config.json of the run (auto-located if omitted)")
    parser.add_argument("--out", required=True, help="output .pt file")
    parser.add_argument("--episodes-per-type", type=int, default=40)
    parser.add_argument("--n-uncontrolled", type=int, nargs="+", default=[1, 2], help="values of K to collect (3s_vs_5z: 1 and/or 2)")
    parser.add_argument("--teammates", nargs="*", default=None, help="subset of teammate keys (default: all in config)")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--uncntrl-path", default=None, help="override base_uncntrl_path (default: src/config/user_info.yaml)")
    parser.add_argument("--sc2path", default=None, help="StarCraft II install dir (default: pysc2 search, /Applications/StarCraft II on macOS)")
    parser.add_argument("--max-episodes", type=int, default=None, help="debug: stop after this many episodes total")
    parser.add_argument("--teammate-seed", type=int, default=None, help="use teammate checkpoints of this seed instead of the trained-against seed (OOD evaluation); rewrites 'seed=<old>' in each agent_path")
    parser.add_argument("--save-every", type=int, default=10, help="write a partial output file every N episodes (0 = off)")
    parser.add_argument("--torch-threads", type=int, default=1, help="torch intra-op threads (1 avoids oversubscription when several collectors share a CPU)")
    return parser.parse_args()


def main():
    cli = parse_args()
    if cli.sc2path:
        os.environ["SC2PATH"] = cli.sc2path

    config_path = cli.config or sacred_config_for_checkpoint(cli.checkpoint)
    if config_path is None or not os.path.exists(config_path):
        sys.exit("Could not find the run's sacred config.json; pass --config")
    config = load_config(config_path)
    print("[collect] config:", config_path)

    user_info_path = os.path.join(REPO, "src", "config", "user_info.yaml")
    user_info = load_yaml(user_info_path) if os.path.exists(user_info_path) else {}
    config["base_uncntrl_path"] = cli.uncntrl_path or user_info.get("base_uncntrl_path", os.path.join(REPO, "uncntrl_agents"))

    # Execution-only overrides. Everything model-related stays as trained.
    config.update({
        "use_cuda": False, "device": "cpu", "batch_size_run": 1, "runner": "episode",
        "render": False, "test_verbose": False, "test_nepisode": 10 ** 9,
        "runner_log_interval": 10 ** 9, "log_discounted_return": False,
        "checkpoint_path": "", "eval_mode": None,
    })
    config.setdefault("env_args", {})["seed"] = cli.seed
    config["seed"] = cli.seed
    config["trained_agents"] = {"agent_0": {"agent_loader": "marie_train_agent_loader", "agent_path": ""}}
    full_uncntrl_order = list(config["uncntrl_agents"].keys())
    teammate_seed_orig = None
    if cli.teammate_seed is not None:
        import re
        for key, agent_cfg in config["uncntrl_agents"].items():
            match = re.search(r"seed=(\d+)", agent_cfg["agent_path"])
            assert match, "no seed in agent_path %s" % agent_cfg["agent_path"]
            teammate_seed_orig = int(match.group(1))
            agent_cfg["agent_path"] = agent_cfg["agent_path"].replace(match.group(0), "seed=%d" % cli.teammate_seed)
            full = os.path.join(config["base_uncntrl_path"], agent_cfg["agent_path"], str(agent_cfg.get("load_step", "best")), "agent.th")
            assert os.path.exists(full), "missing teammate checkpoint %s" % full
            print("[collect] OOD teammate %s -> %s" % (key, agent_cfg["agent_path"]))
    if cli.teammates:
        config["uncntrl_agents"] = {k: v for k, v in config["uncntrl_agents"].items() if k in cli.teammates}
    args = SimpleNamespace(**config)
    args.open_train_or_eval = True

    np.random.seed(cli.seed)
    th.manual_seed(cli.seed)
    if cli.torch_threads:
        th.set_num_threads(cli.torch_threads)

    from utils.logging import Logger, get_logger
    from runners import REGISTRY as runner_registry
    from controllers import REGISTRY as mac_registry
    from components.transforms import OneHot
    from components.episode_buffer import ReplayBuffer

    logger = Logger(get_logger())
    runner = runner_registry["episode"](args=args, logger=logger)
    env_info = runner.get_env_info()
    args.n_agents = env_info["n_agents"]
    args.n_actions = env_info["n_actions"]
    args.state_shape = env_info["state_shape"]
    args.episode_limit = env_info["episode_limit"]
    scheme = {
        "state": {"vshape": env_info["state_shape"]},
        "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
        "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
        "actor_hidden_states": {"vshape": (args.hidden_dim,), "group": "agents"},
        "avail_actions": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.int},
        "reward": {"vshape": (1,)},
        "terminated": {"vshape": (1,), "dtype": th.uint8},
        "trainable_agents": {"vshape": (1,), "group": "agents", "dtype": th.bool},
        "uncontrolled_team_idx": {"vshape": (1,), "dtype": th.long, "episode_const": True},
    }
    groups = {"agents": args.n_agents}
    preprocess = {"actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)])}

    # run.py hands the controller the buffer's scheme, which includes the
    # preprocessed actions_onehot field that the teammate loaders expect.
    buffer_scheme = ReplayBuffer(scheme, groups, 1, env_info["episode_limit"] + 1,
                                 preprocess=preprocess, device="cpu").scheme
    mac = mac_registry["open_train_mac"](buffer_scheme, groups, args)
    runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)
    mac.load_models(cli.checkpoint)
    mac.trained_agent.policy.eval()
    print("[collect] loaded", os.path.join(cli.checkpoint, "agent.th"))

    all_teams = dict(mac.uncontrolled_agent_teams)
    # Canonical labels follow the FULL config order, so files collected with
    # --teammates subsets stay comparable (the controller would number a subset 0..).
    name_to_idx = {name: i for i, name in enumerate(full_uncntrl_order)}
    mac.uncontrolled_team_name_to_idx = {k: name_to_idx[k] for k in all_teams}
    print("[collect] teammate types:", name_to_idx)

    episodes = []
    total = 0
    started = time.time()
    obs_dim = int(np.prod(env_info["obs_shape"])) if not isinstance(env_info["obs_shape"], int) else env_info["obs_shape"]

    def save(path, partial):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        tmp = path + ".tmp"
        th.save({
            "episodes": episodes,
            "team_name_to_idx": name_to_idx,
            "config": config,
            "config_path": os.path.abspath(config_path),
            "checkpoint": os.path.abspath(cli.checkpoint),
            "n_agents": args.n_agents,
            "n_actions": args.n_actions,
            "obs_dim": obs_dim,
            "episode_limit": args.episode_limit,
            "seed": cli.seed,
            "teammate_seed": cli.teammate_seed,
            "teammate_seed_trained_against": teammate_seed_orig,
            "partial": partial,
        }, tmp)
        os.replace(tmp, path)
    for team_name, team in all_teams.items():
        mac.uncontrolled_agent_teams = {team_name: team}
        for k_uncontrolled in cli.n_uncontrolled:
            mac.n_uncontrolled = int(k_uncontrolled)
            for _ in range(cli.episodes_per_type):
                if cli.max_episodes is not None and total >= cli.max_episodes:
                    break
                won_before = runner.test_stats.get("battle_won", 0)
                batch, _ = runner.run(test_mode=True)
                won_after = runner.test_stats.get("battle_won", 0)
                length = int(batch["filled"][0].sum().item())  # frames incl. final
                trainable = batch["trainable_agents"][0, length - 1, :, 0].bool()
                record = {
                    "obs": batch["obs"][0, :length].clone(),                 # [L, n, obs]
                    "actions": batch["actions"][0, :length, :, 0].clone(),   # [L, n]
                    "avail_actions": batch["avail_actions"][0, :length].clone(),
                    "reward": batch["reward"][0, :length - 1, 0].clone(),
                    "terminated": batch["terminated"][0, :length - 1, 0].clone(),
                    "controlled": trainable.clone(),                         # [n] bool
                    "team_idx": int(batch["uncontrolled_team_idx"][0, 0].item()),
                    "team_name": team_name,
                    "n_uncontrolled": int(k_uncontrolled),
                    "return": float(batch["reward"][0, :length - 1, 0].sum().item()),
                    "battle_won": bool(won_after - won_before),
                }
                assert record["team_idx"] == name_to_idx[team_name]
                assert int(trainable.sum()) == args.n_agents - k_uncontrolled
                episodes.append(record)
                total += 1
                if cli.save_every and total % cli.save_every == 0:
                    save(cli.out + ".partial", partial=True)
                if total % 10 == 0:
                    elapsed = time.time() - started
                    print("[collect] %d episodes, %.1fs/episode, last: %s K=%d len=%d return=%.2f won=%s" % (
                        total, elapsed / total, team_name, k_uncontrolled, length - 1,
                        record["return"], record["battle_won"]), flush=True)
    runner.close_env()
    save(cli.out, partial=False)
    print("[collect] saved %d episodes to %s (%.1f min)" % (total, cli.out, (time.time() - started) / 60))


if __name__ == "__main__":
    main()
