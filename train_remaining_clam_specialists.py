#!/usr/bin/env python3
"""Train specialized CLAM policies and their LSTM classifiers for MPE-PP.

By default this handles every complete uncontrolled-agent seed group except
112358 (the group used by the original specialist experiment).  A dry run is
the default; pass ``--run`` to launch jobs sequentially.
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import yaml


AGENT_TYPES = ("ippo", "qmix", "vdn", "mappo", "iql")
ORIGINAL_SEED = 112358
DEFAULT_CLAM_STEP = 20_000_000
DEFAULT_CLASSIFIER_STEP = 40_000_000
GENERATED_CONFIG_DIR = Path("src/config/generated_remaining_clam")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="Run training (default: dry run).")
    parser.add_argument(
        "--only", choices=("all", "clam", "classifier"), default="all",
        help="Train both phases or only one phase.",
    )
    parser.add_argument(
        "--seed", type=int, action="append",
        help="Limit work to a seed; repeat for multiple seeds. May include 112358.",
    )
    parser.add_argument(
        "--agent-type", choices=AGENT_TYPES, action="append",
        help="Limit CLAM training to a type; repeat as needed.",
    )
    parser.add_argument("--uncntrl-root", default="uncntrl_agents")
    parser.add_argument("--results-root", default="naht_results")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--clam-t-max", type=int, default=20_000_000)
    parser.add_argument("--classifier-t-max", type=int, default=40_050_000)
    parser.add_argument("--clam-completion-step", type=int, default=DEFAULT_CLAM_STEP)
    parser.add_argument(
        "--classifier-completion-step", type=int, default=DEFAULT_CLASSIFIER_STEP
    )
    parser.add_argument("--keep-going", action="store_true")
    return parser.parse_args()


def seed_from_name(name: str) -> Optional[int]:
    match = re.search(r"seed=(\d+)_", name)
    return int(match.group(1)) if match else None


def max_checkpoint(model_dir: Path) -> int:
    steps = [
        int(path.name)
        for path in model_dir.iterdir()
        if path.is_dir() and path.name.isdigit()
    ] if model_dir.is_dir() else []
    return max(steps, default=-1)


def complete(model_dir: Path, minimum_step: int) -> bool:
    return (model_dir / "best").exists() and max_checkpoint(model_dir) >= minimum_step


def discover_uncontrolled(root: Path) -> Dict[str, Dict[int, str]]:
    found: Dict[str, Dict[int, str]] = {kind: {} for kind in AGENT_TYPES}
    for kind in AGENT_TYPES:
        model_root = root / "mpe-pp" / kind / "models"
        for model_dir in sorted(model_root.glob(f"{kind}_baseline_seed=*")):
            seed = seed_from_name(model_dir.name)
            if seed is not None:
                found[kind][seed] = (
                    Path("mpe-pp") / kind / "models" / model_dir.name
                ).as_posix()
    return found


def discover_clam(
    results_root: Path, minimum_step: int
) -> Dict[Tuple[str, int], Path]:
    found: Dict[Tuple[str, int], Path] = {}
    for kind in AGENT_TYPES:
        model_root = results_root / "mpe-pp/open_train" / f"clam-vs-{kind}" / "models"
        pattern = f"clam_embed32_specialized_{kind}_seed=*"
        for model_dir in sorted(model_root.glob(pattern)):
            seed = seed_from_name(model_dir.name)
            if seed is None or not complete(model_dir, minimum_step):
                continue
            key = (kind, seed)
            previous = found.get(key)
            if previous is None or max_checkpoint(model_dir) > max_checkpoint(previous):
                found[key] = model_dir
    return found


def discover_classifiers(results_root: Path, minimum_step: int) -> Dict[int, Path]:
    model_root = results_root / "mpe-pp/open_train/clam_lstm_classifier_only/models"
    found: Dict[int, Path] = {}
    for model_dir in sorted(model_root.glob("clam_lstm_classifier_only_*_seed=*")):
        seed = seed_from_name(model_dir.name)
        if seed is None or not complete(model_dir, minimum_step):
            continue
        previous = found.get(seed)
        if previous is None or max_checkpoint(model_dir) > max_checkpoint(previous):
            found[seed] = model_dir
    return found


def common_seeds(models: Dict[str, Dict[int, str]]) -> list[int]:
    return sorted(set.intersection(*(set(models[kind]) for kind in AGENT_TYPES)))


def clam_command(
    args: argparse.Namespace, kind: str, seed: int, agent_path: str
) -> list[str]:
    return [
        args.python,
        "src/main.py",
        f"--config=open/uncntrl_agents/pp_{kind}",
        "--env-config=mpe",
        "--alg-config=mpe/clam",
        f"--seed={seed}",
        f"--label=embed32_specialized_{kind}",
        f"--local_results_path={args.results_root}/mpe-pp/open_train/clam-vs-{kind}",
        "with",
        "env_args.pretrained_wrapper=PretrainedTag",
        "env_args.time_limit=100",
        "env_args.key=mpe:PredatorPrey-v0",
        f"t_max={args.clam_t_max}",
        f"uncntrl_agents.agent_{kind}.agent_path={agent_path}",
        "embed_dim=32",
        "clam_min_crop=16",
        "clam_max_crop=64",
        "clam_mask_ratio=0.2",
    ]


def classifier_config(
    seed: int,
    uncontrolled: Dict[str, Dict[int, str]],
    specialists: Dict[Tuple[str, int], Path],
    classifier_t_max: int,
) -> Path:
    template = Path("src/config/open/clam_lstm_classifier_only_train_pp.yaml")
    with template.open() as stream:
        config = yaml.safe_load(stream)

    config["t_max"] = classifier_t_max
    config["label"] = "classifier_lstm_clam_specialists"
    trained = config["trained_agents"]["agent_0"]
    trained["base_path"] = "."
    trained["teammate_types"] = [
        {
            "name": kind,
            "agent_loader": "clam_eval_agent_loader",
            "agent_path": specialists[(kind, seed)].as_posix(),
            "load_step": "best",
            "test_mode": True,
        }
        for kind in AGENT_TYPES
    ]
    config["uncntrl_agents"] = {
        f"agent_{kind}": {
            "agent_loader": "rnn_eval_agent_loader",
            "agent_path": uncontrolled[kind][seed],
            "load_step": "best",
            "n_agents_to_populate": 3,
            "test_mode": True,
        }
        for kind in AGENT_TYPES
    }

    output = GENERATED_CONFIG_DIR / f"lstm_classifier_uncseed_{seed}.yaml"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as stream:
        yaml.safe_dump(config, stream, sort_keys=False)
    return output


def classifier_command(args: argparse.Namespace, config: Path, seed: int) -> list[str]:
    config_name = config.with_suffix("").relative_to("src/config").as_posix()
    return [
        args.python,
        "src/main.py",
        f"--config={config_name}",
        "--env-config=gymma",
        "--alg-config=mpe/poam_type_classifier",
        f"--seed={seed}",
    ]


def run_command(command: list[str], keep_going: bool) -> bool:
    print("  " + shlex.join(command), flush=True)
    result = subprocess.run(command)
    if result.returncode == 0:
        return True
    print(f"Command failed with exit code {result.returncode}.", file=sys.stderr)
    if not keep_going:
        raise SystemExit(result.returncode)
    return False


def validate_args(args: argparse.Namespace) -> None:
    for name in ("clam_t_max", "classifier_t_max", "clam_completion_step", "classifier_completion_step"):
        if getattr(args, name) <= 0:
            raise SystemExit(f"--{name.replace('_', '-')} must be positive")
    if not Path("src/config/open/clam_lstm_classifier_only_train_pp.yaml").is_file():
        raise SystemExit("Missing CLAM LSTM classifier template config")


def main() -> int:
    args = parse_args()
    validate_args(args)
    uncontrolled = discover_uncontrolled(Path(args.uncntrl_root))
    available = common_seeds(uncontrolled)
    seeds = sorted(set(args.seed)) if args.seed else [s for s in available if s != ORIGINAL_SEED]
    missing = sorted(set(seeds) - set(available))
    if missing:
        raise SystemExit(f"Seeds without all five uncontrolled checkpoints: {missing}")

    selected_types: Iterable[str] = args.agent_type or AGENT_TYPES
    specialists = discover_clam(Path(args.results_root), args.clam_completion_step)
    classifiers = discover_classifiers(Path(args.results_root), args.classifier_completion_step)

    print(f"Complete uncontrolled seed groups: {available}")
    print(f"Selected seeds: {seeds}")
    print("Mode: " + ("run" if args.run else "dry run"))

    for seed in seeds:
        print(f"\n=== Seed {seed} ===")
        if args.only in ("all", "clam"):
            for kind in selected_types:
                if (kind, seed) in specialists:
                    print(f"SKIP complete specialized CLAM vs {kind}")
                    continue
                command = clam_command(args, kind, seed, uncontrolled[kind][seed])
                if args.run:
                    print(f"RUN specialized CLAM vs {kind}")
                    run_command(command, args.keep_going)
                else:
                    print(f"PLAN specialized CLAM vs {kind}")
                    print("  " + shlex.join(command))

        if args.only in ("all", "classifier"):
            if seed in classifiers:
                print("SKIP complete LSTM classifier")
                continue
            # Training creates timestamped paths, so rediscover after CLAM jobs.
            specialists = discover_clam(Path(args.results_root), args.clam_completion_step)
            absent = [kind for kind in AGENT_TYPES if (kind, seed) not in specialists]
            if absent:
                phase = "will wait for" if not args.run and args.only == "all" else "blocked by"
                print(f"PLAN LSTM classifier ({phase}: {', '.join(absent)})")
                continue
            config = classifier_config(seed, uncontrolled, specialists, args.classifier_t_max)
            command = classifier_command(args, config, seed)
            if args.run:
                print("RUN LSTM classifier")
                run_command(command, args.keep_going)
            else:
                print("PLAN LSTM classifier")
                print("  " + shlex.join(command))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
