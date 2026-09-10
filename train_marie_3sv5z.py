#!/usr/bin/env python3
"""Launch MARIE on SMAC 3s_vs_5z through the EPyMARL runtime."""

import argparse
import os
from pathlib import Path
import subprocess
import sys


TNTEAM_ROOT = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend", choices=("epymarl", "tnteam-port"), default="epymarl",
        help="Both names run the self-contained tnteam implementation.",
    )
    parser.add_argument(
        "--seed", type=int, action="append",
        help="Original seed index; seed 1 becomes effective seed 123.",
    )
    parser.add_argument("--steps", type=int, default=200_000)
    parser.add_argument("--eval-episodes", type=int, default=8)
    parser.add_argument("--eval-workers", type=int, default=1)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--keep-going", action="store_true")
    return parser.parse_args()


def port_command(python, seed, steps, eval_episodes):
    effective_seed = 23 + 100 * seed
    return [
        python, "src/main.py", "--config=default/default_3sv5z",
        "--env-config=sc2",
        "--alg-config=sc2/marie",
        f"--seed={effective_seed}", "with", "env_args.map_name=3s_vs_5z",
        f"t_max={steps}", f"test_nepisode={eval_episodes}",
        "local_results_path=3sv5z/full_control/marie",
    ]


def main():
    args = parse_args()
    seeds = args.seed or [1]
    if args.eval_episodes < 1 or args.eval_workers < 1:
        raise SystemExit("evaluation episodes/workers must be positive")
    environment = os.environ.copy()
    environment.setdefault(
        "SC2PATH", str(TNTEAM_ROOT / "3rdparty" / "StarCraftII")
    )
    for seed in seeds:
        command = port_command(
            args.python, seed, args.steps, args.eval_episodes
        )
        cwd = TNTEAM_ROOT
        print(" ".join(command), flush=True)
        if args.dry_run:
            continue
        result = subprocess.run(command, cwd=cwd, env=environment)
        if result.returncode and not args.keep_going:
            return result.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
