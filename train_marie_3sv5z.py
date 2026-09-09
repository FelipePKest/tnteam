#!/usr/bin/env python3
"""Launch MARIE on SMAC 3s_vs_5z through the EPyMARL runtime."""

import argparse
import os
from pathlib import Path
import subprocess
import sys


TNTEAM_ROOT = Path(__file__).resolve().parent
REFERENCE_ROOT = Path(
    os.environ.get("MARIE_REFERENCE_ROOT", "/home/jupyter-jphuser3/MARIE")
).resolve()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend", choices=("epymarl", "original", "tnteam-port"), default="epymarl"
    )
    parser.add_argument(
        "--seed", type=int, action="append",
        help="Original seed index; seed 1 becomes effective seed 123.",
    )
    parser.add_argument("--steps", type=int, default=200_000)
    parser.add_argument("--eval-episodes", type=int, default=8)
    parser.add_argument("--eval-workers", type=int, default=1)
    parser.add_argument("--mode", default="online")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--keep-going", action="store_true")
    return parser.parse_args()


def original_command(python, seed, steps, mode):
    return [
        python, str(REFERENCE_ROOT / "train.py"),
        "--n_workers", "1", "--env", "starcraft",
        "--env_name", "3s_vs_5z", "--seed", str(seed),
        "--steps", str(steps), "--mode", mode,
        "--tokenizer", "vq", "--decay", "0.8",
        "--temperature", "1.0", "--sample_temp", "inf", "--ce_for_av",
    ]


def port_command(python, seed, steps, eval_episodes, reference=False):
    effective_seed = 23 + 100 * seed
    return [
        python, "src/main.py", "--config=default/default_3sv5z",
        "--env-config=sc2",
        "--alg-config=sc2/marie_reference" if reference else "--alg-config=sc2/marie",
        f"--seed={effective_seed}", "with", "env_args.map_name=3s_vs_5z",
        f"t_max={steps}", f"test_nepisode={eval_episodes}",
        "local_results_path=" + (
            "naht_results/3sv5z/full_control/marie_reference_epymarl"
            if reference else "naht_results/3sv5z/full_control/marie_port"
        ),
    ]


def main():
    args = parse_args()
    seeds = args.seed or [1]
    if args.eval_episodes < 1 or args.eval_workers < 1:
        raise SystemExit("evaluation episodes/workers must be positive")
    if args.backend == "original" and not (REFERENCE_ROOT / "train.py").is_file():
        raise SystemExit(f"reference MARIE checkout not found at {REFERENCE_ROOT}")
    environment = os.environ.copy()
    # The reference EnvConfigs module eagerly imports its optional MuJoCo
    # backend even for StarCraft. Supplying its installed library directory
    # lets that unused import complete without changing any MARIE code.
    mujoco_library = str(Path.home() / ".mujoco" / "mujoco210" / "bin")
    current_library_path = environment.get("LD_LIBRARY_PATH", "")
    environment["LD_LIBRARY_PATH"] = ":".join(part for part in (
        current_library_path, mujoco_library, "/usr/lib/nvidia"
    ) if part)
    environment["MARIE_EVAL_EPISODES"] = str(args.eval_episodes)
    environment["MARIE_EVAL_WORKERS"] = str(
        min(args.eval_workers, args.eval_episodes)
    )
    environment.setdefault(
        "SC2PATH", str(TNTEAM_ROOT / "3rdparty" / "StarCraftII")
    )
    for seed in seeds:
        if args.backend == "original":
            command = original_command(args.python, seed, args.steps, args.mode)
            cwd = REFERENCE_ROOT
        else:
            command = port_command(
                args.python, seed, args.steps, args.eval_episodes,
                reference=args.backend == "epymarl",
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
