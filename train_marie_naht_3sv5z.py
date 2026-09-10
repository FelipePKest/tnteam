#!/usr/bin/env python3
"""Launch tnteam's self-contained MARIE learner in the NAHT setting."""

import argparse
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--steps", type=int, default=200_000)
    parser.add_argument("--eval-episodes", type=int, default=8)
    parser.add_argument("--batch-size-run", type=int, default=1)
    parser.add_argument("--uncontrolled-seed", type=int)
    parser.add_argument("--n-uncontrolled", type=int, choices=(1, 2))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    command = [
        sys.executable, "src/main.py",
        "--config=open/open_train_3sv5z",
        "--env-config=sc2",
        "--alg-config=sc2/marie",
        f"--seed={args.seed}", "with",
        "name=marie_naht",
        "env_args.map_name=3s_vs_5z",
        f"t_max={args.steps}",
        f"test_nepisode={args.eval_episodes}",
        f"batch_size_run={args.batch_size_run}",
        "trained_agents.agent_0.agent_loader=marie_train_agent_loader",
        "trained_agents.agent_0.agent_path=",
        f"local_results_path=3sv5z/open_train/marie-naht-bsr{args.batch_size_run}-seed{args.seed}",
    ]
    if args.uncontrolled_seed is not None:
        teammate_runs = {
            "agent_ippo": ("ippo", "22-51-47"),
            "agent_qmix": ("qmix", "22-49-07"),
            "agent_vdn": ("vdn", "22-49-51"),
            "agent_mappo": ("mappo", "22-52-21"),
            "agent_iql": ("iql", "22-50-45"),
        }
        for agent_key, (algorithm, timestamp) in teammate_runs.items():
            command.append(
                f"uncntrl_agents.{agent_key}.agent_path="
                f"3sv5z/{algorithm}/models/{algorithm}_baseline_seed="
                f"{args.uncontrolled_seed}_02-29-{timestamp}"
            )
    if args.n_uncontrolled is not None:
        command.append(f"n_uncontrolled={args.n_uncontrolled}")
    print(" ".join(command), flush=True)
    if args.dry_run:
        return 0
    environment = os.environ.copy()
    environment.setdefault("SC2PATH", str(ROOT / "3rdparty" / "StarCraftII"))
    return subprocess.run(command, cwd=ROOT, env=environment).returncode


if __name__ == "__main__":
    raise SystemExit(main())
