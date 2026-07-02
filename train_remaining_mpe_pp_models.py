#!/usr/bin/env python3
"""Train missing MPE-PP POAM-vs and LSTM classifier models.

The script discovers uncontrolled-agent seeds from ``uncntrl_agents/mpe-pp``
and completed training runs from one or more result roots. It then generates
temporary configs for the missing jobs and, when ``--run`` is passed, launches
them sequentially.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import yaml


AGENT_TYPES = ["ippo", "qmix", "vdn", "mappo", "iql"]
DEFAULT_COMPLETION_STEP = 20_000_000
GENERATED_CONFIG_DIR = Path("src/config/generated_remaining_mpe_pp")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate and optionally run all missing MPE-PP POAM-vs and "
            "LSTM classifier training jobs."
        )
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Execute jobs. Without this flag, only prints the planned jobs.",
    )
    parser.add_argument(
        "--only",
        choices=["all", "poam", "classifier"],
        default="all",
        help="Restrict which job family to train.",
    )
    parser.add_argument(
        "--agent-type",
        choices=AGENT_TYPES,
        action="append",
        help="Restrict POAM-vs jobs to one or more uncontrolled agent types.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        action="append",
        help="Restrict jobs to one or more uncontrolled-agent seeds.",
    )
    parser.add_argument(
        "--results-root",
        action="append",
        default=["remote-results", "naht_results"],
        help=(
            "Result root to scan for completed jobs. Can be repeated. "
            "Defaults to remote-results and naht_results."
        ),
    )
    parser.add_argument(
        "--uncntrl-root",
        default="uncntrl_agents",
        help="Root containing uncontrolled-agent checkpoints.",
    )
    parser.add_argument(
        "--completion-step",
        type=int,
        default=DEFAULT_COMPLETION_STEP,
        help="Minimum numeric checkpoint step required to treat a run as complete.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used to launch src/main.py. Defaults to this interpreter.",
    )
    parser.add_argument(
        "--manifest",
        default="remaining_mpe_pp_jobs.json",
        help="Path to write a JSON manifest of planned jobs.",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="Continue after a failed job instead of stopping immediately.",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> dict:
    with path.open("r") as f:
        return yaml.safe_load(f)


def write_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def model_seed(path_name: str) -> Optional[int]:
    match = re.search(r"seed=(\d+)_", path_name)
    return int(match.group(1)) if match else None


def max_numeric_checkpoint(model_dir: Path) -> int:
    if not model_dir.exists():
        return -1
    steps = [
        int(child.name)
        for child in model_dir.iterdir()
        if child.is_dir() and child.name.isdigit()
    ]
    return max(steps) if steps else -1


def is_complete_model(model_dir: Path, completion_step: int) -> bool:
    return (model_dir / "best").exists() and max_numeric_checkpoint(model_dir) >= completion_step


def sacred_config_for_model(model_dir: Path) -> Optional[dict]:
    try:
        relative = model_dir.relative_to(model_dir.parents[2])
    except ValueError:
        return None
    sacred_path = model_dir.parents[2] / relative.parent.parent / "sacred" / model_dir.name / "1" / "config.json"
    if not sacred_path.exists():
        return None
    with sacred_path.open("r") as f:
        return json.load(f)


def poam_config_matches(model_dir: Path, agent_type: str, seed: int) -> bool:
    config = sacred_config_for_model(model_dir)
    if config is None:
        return False
    uncntrl_agents = config.get("uncntrl_agents", {})
    expected_fragment = f"mpe-pp/{agent_type}/models/{agent_type}_baseline_seed={seed}_"
    return any(
        agent_cfg.get("agent_path", "").startswith(expected_fragment)
        for agent_cfg in uncntrl_agents.values()
    )


def classifier_config_matches(model_dir: Path, seed: int) -> bool:
    config = sacred_config_for_model(model_dir)
    if config is None:
        return False

    uncntrl_agents = config.get("uncntrl_agents", {})
    for agent_type in AGENT_TYPES:
        expected = f"mpe-pp/{agent_type}/models/{agent_type}_baseline_seed={seed}_"
        if not any(
            agent_cfg.get("agent_path", "").startswith(expected)
            for key, agent_cfg in uncntrl_agents.items()
            if key == f"agent_{agent_type}"
        ):
            return False

    teammate_types = (
        config.get("trained_agents", {})
        .get("agent_0", {})
        .get("teammate_types", [])
    )
    by_name = {agent_cfg.get("name"): agent_cfg for agent_cfg in teammate_types}
    for agent_type in AGENT_TYPES:
        agent_path = by_name.get(agent_type, {}).get("agent_path", "")
        expected = f"poam-vs-{agent_type}/models/poam_baseline_seed={seed}_"
        if expected not in agent_path:
            return False
    return True


def discover_uncntrl_models(uncntrl_root: Path) -> Dict[str, Dict[int, str]]:
    models: Dict[str, Dict[int, str]] = {agent_type: {} for agent_type in AGENT_TYPES}
    for agent_type in AGENT_TYPES:
        model_root = uncntrl_root / "mpe-pp" / agent_type / "models"
        for model_dir in sorted(model_root.glob(f"{agent_type}_baseline_seed=*")):
            seed = model_seed(model_dir.name)
            if seed is None:
                continue
            rel_path = Path("mpe-pp") / agent_type / "models" / model_dir.name
            models[agent_type][seed] = rel_path.as_posix()
    return models


def discover_completed_poam(
    result_roots: Iterable[Path],
    completion_step: int,
) -> Dict[Tuple[str, int], Path]:
    completed: Dict[Tuple[str, int], Path] = {}
    for root in result_roots:
        for agent_type in AGENT_TYPES:
            model_root = root / "mpe-pp" / "open_train" / f"poam-vs-{agent_type}" / "models"
            for model_dir in sorted(model_root.glob("poam_baseline_seed=*")):
                seed = model_seed(model_dir.name)
                if (
                    seed is None
                    or not is_complete_model(model_dir, completion_step)
                    or not poam_config_matches(model_dir, agent_type, seed)
                ):
                    continue
                key = (agent_type, seed)
                current = completed.get(key)
                if current is None or max_numeric_checkpoint(model_dir) >= max_numeric_checkpoint(current):
                    completed[key] = model_dir
    return completed


def discover_completed_classifiers(
    result_roots: Iterable[Path],
    completion_step: int,
) -> Dict[int, Path]:
    completed: Dict[int, Path] = {}
    for root in result_roots:
        model_root = root / "mpe-pp" / "open_train" / "poam_lstm_classifier_only" / "models"
        for model_dir in sorted(model_root.glob("poam_lstm_classifier_only_classifier_lstm_only_seed=*")):
            seed = model_seed(model_dir.name)
            if (
                seed is None
                or not is_complete_model(model_dir, completion_step)
                or not classifier_config_matches(model_dir, seed)
            ):
                continue
            current = completed.get(seed)
            if current is None or max_numeric_checkpoint(model_dir) >= max_numeric_checkpoint(current):
                completed[seed] = model_dir
    return completed


def restrict_seeds(all_seeds: Iterable[int], requested: Optional[List[int]]) -> List[int]:
    seeds = sorted(set(all_seeds))
    if requested is None:
        return seeds
    requested_set = set(requested)
    return [seed for seed in seeds if seed in requested_set]


def config_name(config_path: Path) -> str:
    rel = config_path.with_suffix("").relative_to(Path("src/config"))
    return rel.as_posix()


def build_poam_config(agent_type: str, seed: int, agent_path: str) -> Path:
    base = load_yaml(Path("src/config/open/uncntrl_agents") / f"pp_{agent_type}.yaml")
    base["uncntrl_agents"] = {
        f"agent_{agent_type}": {
            "agent_loader": "rnn_eval_agent_loader",
            "agent_path": agent_path,
            "load_step": "best",
            "n_agents_to_populate": 3,
            "test_mode": True,
        }
    }
    base["local_results_path"] = f"mpe-pp/open_train/poam-vs-{agent_type}"
    base["label"] = "baseline"
    out_path = GENERATED_CONFIG_DIR / f"poam_vs_{agent_type}_uncseed_{seed}.yaml"
    write_yaml(out_path, base)
    return out_path


def build_classifier_config(
    seed: int,
    uncntrl_models: Dict[str, Dict[int, str]],
    poam_models: Dict[Tuple[str, int], Path],
) -> Path:
    base = load_yaml(Path("src/config/open/classifier_lstm_only_train_pp.yaml"))
    base["local_results_path"] = "mpe-pp/open_train/poam_lstm_classifier_only"
    base["label"] = "classifier_lstm_only"

    # Use direct model paths so the classifier can mix remote-results and
    # newly generated naht_results POAM-vs checkpoints for the same seed.
    trained_agent = base["trained_agents"]["agent_0"]
    trained_agent["base_path"] = "."
    teammate_types = []
    for agent_type in AGENT_TYPES:
        teammate_types.append(
            {
                "name": agent_type,
                "agent_loader": "poam_eval_agent_loader",
                "agent_path": poam_models[(agent_type, seed)].as_posix(),
                "load_step": "best",
                "test_mode": True,
            }
        )
    trained_agent["teammate_types"] = teammate_types

    base["uncntrl_agents"] = {}
    for agent_type in AGENT_TYPES:
        base["uncntrl_agents"][f"agent_{agent_type}"] = {
            "agent_loader": "rnn_eval_agent_loader",
            "agent_path": uncntrl_models[agent_type][seed],
            "load_step": "best",
            "n_agents_to_populate": 3,
            "test_mode": True,
        }

    out_path = GENERATED_CONFIG_DIR / f"lstm_classifier_uncseed_{seed}.yaml"
    write_yaml(out_path, base)
    return out_path


def command_for_poam(python_exe: str, config_path: Path, seed: int) -> List[str]:
    return [
        python_exe,
        "src/main.py",
        f"--config={config_name(config_path)}",
        "--env-config=mpe",
        "--alg-config=mpe/poam",
        f"--seed={seed}",
        "with",
        "env_args.key=mpe:PredatorPrey-v0",
        "env_args.pretrained_wrapper=PretrainedTag",
        "env_args.time_limit=100",
    ]


def command_for_classifier(python_exe: str, config_path: Path, seed: int) -> List[str]:
    return [
        python_exe,
        "src/main.py",
        f"--config={config_name(config_path)}",
        "--env-config=gymma",
        "--alg-config=mpe/poam_type_classifier",
        f"--seed={seed}",
    ]


def shell_join(command: List[str]) -> str:
    return " ".join(subprocess.list2cmdline([part]) for part in command)


def run_job(job: dict, keep_going: bool) -> bool:
    print(f"\n=== Running {job['kind']} job: {job['id']} ===", flush=True)
    print(shell_join(job["command"]), flush=True)
    result = subprocess.run(job["command"])
    if result.returncode == 0:
        return True
    print(f"Job failed with exit code {result.returncode}: {job['id']}", flush=True)
    if not keep_going:
        raise SystemExit(result.returncode)
    return False


def plan_poam_jobs(
    args: argparse.Namespace,
    uncntrl_models: Dict[str, Dict[int, str]],
    completed_poam: Dict[Tuple[str, int], Path],
) -> List[dict]:
    selected_types = args.agent_type or AGENT_TYPES
    all_seeds = set()
    for agent_type in selected_types:
        all_seeds.update(uncntrl_models[agent_type].keys())
    seeds = restrict_seeds(all_seeds, args.seed)

    jobs = []
    for seed in seeds:
        for agent_type in selected_types:
            agent_path = uncntrl_models[agent_type].get(seed)
            if agent_path is None:
                continue
            if (agent_type, seed) in completed_poam:
                continue
            config_path = build_poam_config(agent_type, seed, agent_path)
            jobs.append(
                {
                    "id": f"poam-vs-{agent_type}:uncontrolled-seed={seed}",
                    "kind": "poam",
                    "agent_type": agent_type,
                    "seed": seed,
                    "config": config_path.as_posix(),
                    "command": command_for_poam(args.python, config_path, seed),
                }
            )
    return jobs


def plan_classifier_jobs(
    args: argparse.Namespace,
    uncntrl_models: Dict[str, Dict[int, str]],
    completed_poam: Dict[Tuple[str, int], Path],
    completed_classifiers: Dict[int, Path],
) -> Tuple[List[dict], List[dict]]:
    shared_seeds = set.intersection(
        *(set(uncntrl_models[agent_type].keys()) for agent_type in AGENT_TYPES)
    )
    seeds = restrict_seeds(shared_seeds, args.seed)

    jobs = []
    blocked = []
    for seed in seeds:
        if seed in completed_classifiers:
            continue
        missing_poam = [
            agent_type
            for agent_type in AGENT_TYPES
            if (agent_type, seed) not in completed_poam
        ]
        if missing_poam:
            blocked.append(
                {
                    "id": f"lstm-classifier:uncontrolled-seed={seed}",
                    "seed": seed,
                    "missing_poam": missing_poam,
                }
            )
            continue
        config_path = build_classifier_config(seed, uncntrl_models, completed_poam)
        jobs.append(
            {
                "id": f"lstm-classifier:uncontrolled-seed={seed}",
                "kind": "classifier",
                "seed": seed,
                "config": config_path.as_posix(),
                "command": command_for_classifier(args.python, config_path, seed),
            }
        )
    return jobs, blocked


def write_manifest(path: Path, jobs: List[dict], blocked: List[dict]) -> None:
    serializable = deepcopy(jobs)
    for job in serializable:
        job["command"] = shell_join(job["command"])
    path.write_text(json.dumps({"jobs": serializable, "blocked": blocked}, indent=2))


def main() -> int:
    args = parse_args()
    result_roots = [Path(root) for root in args.results_root]
    uncntrl_root = Path(args.uncntrl_root)

    uncntrl_models = discover_uncntrl_models(uncntrl_root)
    completed_poam = discover_completed_poam(result_roots, args.completion_step)
    completed_classifiers = discover_completed_classifiers(result_roots, args.completion_step)

    jobs: List[dict] = []
    blocked: List[dict] = []

    if args.only in ("all", "poam"):
        jobs.extend(plan_poam_jobs(args, uncntrl_models, completed_poam))

    if args.only in ("all", "classifier"):
        classifier_jobs, blocked = plan_classifier_jobs(
            args,
            uncntrl_models,
            completed_poam,
            completed_classifiers,
        )
        jobs.extend(classifier_jobs)

    write_manifest(Path(args.manifest), jobs, blocked)

    print(f"Discovered uncontrolled seeds by type:")
    for agent_type in AGENT_TYPES:
        print(f"  {agent_type}: {sorted(uncntrl_models[agent_type])}")
    print(f"\nPlanned jobs: {len(jobs)}")
    for job in jobs:
        print(f"  - {job['id']}")
    if blocked:
        print("\nBlocked classifier jobs waiting for POAM-vs models:")
        for item in blocked:
            print(f"  - seed={item['seed']}: missing {', '.join(item['missing_poam'])}")
    print(f"\nManifest written to {args.manifest}")

    if not args.run:
        print("\nDry run only. Re-run with --run to execute the planned jobs.")
        return 0

    if args.only == "all":
        # Execute POAM jobs first, then rediscover and execute classifier jobs
        # that may have become unblocked by the newly trained policies.
        poam_jobs = [job for job in jobs if job["kind"] == "poam"]
        for job in poam_jobs:
            run_job(job, keep_going=args.keep_going)

        completed_poam = discover_completed_poam(result_roots, args.completion_step)
        completed_classifiers = discover_completed_classifiers(result_roots, args.completion_step)
        classifier_jobs, blocked = plan_classifier_jobs(
            args,
            uncntrl_models,
            completed_poam,
            completed_classifiers,
        )
        for job in classifier_jobs:
            run_job(job, keep_going=args.keep_going)
        if blocked:
            print("\nSome classifier jobs remain blocked:")
            for item in blocked:
                print(f"  - seed={item['seed']}: missing {', '.join(item['missing_poam'])}")
        return 0

    for job in jobs:
        run_job(job, keep_going=args.keep_going)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

