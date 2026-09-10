#!/usr/bin/env python3
"""Run POAM on 3s_vs_5z at MARIE's measured policy update/data ratio."""

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
    parser.add_argument("--updates-per-step", type=float, default=0.77)
    parser.add_argument("--batch-size-run", type=int, default=1)
    parser.add_argument("--uncontrolled-seed", type=int)
    parser.add_argument("--checkpoint-interval", type=int, default=10_000)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--load-step", default="last")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    # Reference MARIE performed about 907 learner cycles in 147,795 steps.
    # A mature cycle makes about 125 actor optimizer steps (5 imagined
    # rollouts * 5 PPO epochs * ~5 minibatches), or ~0.77 actor updates per
    # real transition. POAM dynamically scales its epochs from each episode's
    # actual unpadded length to maintain this ratio.
    command = [
        sys.executable,
        "src/main.py",
        "--config=open/open_train_3sv5z",
        "--env-config=sc2",
        "--alg-config=sc2/poam",
        f"--seed={args.seed}",
        "with",
        "env_args.map_name=3s_vs_5z",
        f"t_max={args.steps}",
        f"test_nepisode={args.eval_episodes}",
        "test_interval=10000",
        "log_interval=10000",
        "runner_log_interval=10000",
        "learner_log_interval=10000",
        f"batch_size_run={args.batch_size_run}",
        f"batch_size={args.batch_size_run}",
        f"buffer_size={args.batch_size_run}",
        "epochs=5",
        f"poam_actor_updates_per_env_step={args.updates_per_step}",
        "n_minibatch=1",
        "n_ed_minibatch=1",
        f"local_results_path=3sv5z/open_train/poam-marie-utd-bsr{args.batch_size_run}-seed{args.seed}",
        f"save_model_interval={args.checkpoint_interval}",
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
    if args.checkpoint:
        command.extend([
            f"checkpoint_path={args.checkpoint}",
            f"load_step={args.load_step}",
            "resume_t_env_from_checkpoint=True",
        ])
    print(" ".join(command), flush=True)
    if args.dry_run:
        return 0

    environment = os.environ.copy()
    environment.setdefault("SC2PATH", str(ROOT / "3rdparty" / "StarCraftII"))
    return subprocess.run(command, cwd=ROOT, env=environment).returncode


if __name__ == "__main__":
    raise SystemExit(main())
