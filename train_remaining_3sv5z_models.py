#!/usr/bin/env python3
"""Generate and optionally run missing 3sv5z POAM training jobs.

This mirrors train_remaining_mpe_pp_models.py but targets the SC2 3s_vs_5z
environment and configuration templates under src/config/open/uncntrl_agents/3sv5z_*.yaml.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import yaml


AGENT_TYPES = ["ippo", "qmix", "vdn", "mappo", "iql"]
GENERATED_CONFIG_DIR = Path("src/config/generated_remaining_3sv5z")
DEFAULT_COMPLETION_STEP = 20_000_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate/run missing 3sv5z POAM jobs")
    parser.add_argument("--run", action="store_true", help="Execute planned jobs")
    parser.add_argument("--agent-type", choices=AGENT_TYPES, action="append")
    parser.add_argument("--seed", type=int, action="append")
    parser.add_argument("--results-root", action="append", default=["remote-results", "naht_results"])
    parser.add_argument("--uncntrl-root", default="uncntrl_agents")
    parser.add_argument("--completion-step", type=int, default=DEFAULT_COMPLETION_STEP)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--manifest", default="remaining_3sv5z_jobs.json")
    parser.add_argument("--keep-going", action="store_true")
    return parser.parse_args()


def load_yaml(path: Path) -> dict:
    with path.open("r") as f:
        return yaml.safe_load(f)


def write_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def model_seed(path_name: str) -> Optional[int]:
    m = re.search(r"seed=(\d+)_", path_name)
    return int(m.group(1)) if m else None


def max_numeric_checkpoint(model_dir: Path) -> int:
    if not model_dir.exists():
        return -1
    steps = [int(c.name) for c in model_dir.iterdir() if c.is_dir() and c.name.isdigit()]
    return max(steps) if steps else -1


def is_complete_model(model_dir: Path, completion_step: int) -> bool:
    return (model_dir / "best").exists() and max_numeric_checkpoint(model_dir) >= completion_step


def sacred_config_for_model(model_dir: Path) -> Optional[dict]:
    sacred_path = model_dir.parent.parent / "sacred" / model_dir.name / "1" / "config.json"
    if not sacred_path.exists():
        return None
    with sacred_path.open("r") as f:
        return json.load(f)


def discover_uncntrl_models(uncntrl_root: Path) -> Dict[str, Dict[int, str]]:
    models: Dict[str, Dict[int, str]] = {t: {} for t in AGENT_TYPES}
    for t in AGENT_TYPES:
        model_root = uncntrl_root / "3sv5z" / t / "models"
        if not model_root.exists():
            continue
        for model_dir in sorted(model_root.glob(f"{t}_baseline_seed=*")):
            seed = model_seed(model_dir.name)
            if seed is None:
                continue
            rel = Path("3sv5z") / t / "models" / model_dir.name
            models[t][seed] = rel.as_posix()
    return models


def poam_config_matches(model_dir: Path, agent_type: str, seed: int) -> bool:
    config = sacred_config_for_model(model_dir)
    if config is None:
        return False
    uncntrl = config.get("uncntrl_agents", {})
    expected = f"3sv5z/{agent_type}/models/{agent_type}_baseline_seed={seed}_"
    return any(c.get("agent_path", "").startswith(expected) for c in uncntrl.values())


def discover_completed_poam(result_roots: Iterable[Path], completion_step: int) -> Dict[Tuple[str, int], Path]:
    completed: Dict[Tuple[str, int], Path] = {}
    for root in result_roots:
        for agent_type in AGENT_TYPES:
            model_root = root / "3sv5z" / "open_train" / "poam-pqvmq_open" / "models"
            if not model_root.exists():
                continue
            for model_dir in sorted(model_root.glob("poam_baseline_seed=*")):
                seed = model_seed(model_dir.name)
                if seed is None or not is_complete_model(model_dir, completion_step) or not poam_config_matches(model_dir, agent_type, seed):
                    continue
                key = (agent_type, seed)
                cur = completed.get(key)
                if cur is None or max_numeric_checkpoint(model_dir) >= max_numeric_checkpoint(cur):
                    completed[key] = model_dir
    return completed


def restrict_seeds(all_seeds: Iterable[int], requested: Optional[List[int]]) -> List[int]:
    s = sorted(set(all_seeds))
    if requested is None:
        return s
    req = set(requested)
    return [x for x in s if x in req]


def config_name(path: Path) -> str:
    return path.with_suffix("").relative_to(Path("src/config")).as_posix()


def build_poam_config(agent_type: str, seed: int, agent_path: str) -> Path:
    base = load_yaml(Path("src/config/open/uncntrl_agents") / f"3sv5z_{agent_type}.yaml")
    base["uncntrl_agents"] = {
        f"agent_{agent_type}": {
            "agent_loader": "rnn_eval_agent_loader",
            "agent_path": agent_path,
            "load_step": "best",
            "n_agents_to_populate": 3,
            "test_mode": True,
        }
    }
    base["local_results_path"] = f"3sv5z/open_train/poam-pqvmq_open"
    base["label"] = "baseline"
    out = GENERATED_CONFIG_DIR / f"poam_vs_{agent_type}_uncseed_{seed}.yaml"
    write_yaml(out, base)
    return out


def command_for_poam(python_exe: str, config_path: Path, seed: int) -> List[str]:
    return [
        python_exe,
        "src/main.py",
        f"--config={config_name(config_path)}",
        "--env-config=sc2",
        "--alg-config=sc2/poam",
        f"--seed={seed}",
        "with",
        "env_args.map_name=3s_vs_5z",
    ]


def shell_join(command: List[str]) -> str:
    return " ".join(subprocess.list2cmdline([p]) for p in command)


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


def plan_poam_jobs(args: argparse.Namespace, uncntrl_models: Dict[str, Dict[int, str]], completed_poam: Dict[Tuple[str, int], Path]) -> List[dict]:
    selected = args.agent_type or AGENT_TYPES
    all_seeds = set()
    for t in selected:
        all_seeds.update(uncntrl_models.get(t, {}).keys())
    seeds = restrict_seeds(all_seeds, args.seed)

    jobs = []
    for seed in seeds:
        for t in selected:
            agent_path = uncntrl_models.get(t, {}).get(seed)
            if agent_path is None:
                continue
            if (t, seed) in completed_poam:
                continue
            cfg = build_poam_config(t, seed, agent_path)
            jobs.append({
                "id": f"poam-vs-{t}:uncontrolled-seed={seed}",
                "kind": "poam",
                "agent_type": t,
                "seed": seed,
                "config": cfg.as_posix(),
                "command": command_for_poam(args.python, cfg, seed),
            })
    return jobs


def write_manifest(path: Path, jobs: List[dict]) -> None:
    serial = deepcopy(jobs)
    for j in serial:
        j["command"] = shell_join(j["command"])
    path.write_text(json.dumps({"jobs": serial}, indent=2))


def main() -> int:
    args = parse_args()
    result_roots = [Path(r) for r in args.results_root]
    uncntrl_root = Path(args.uncntrl_root)

    uncntrl_models = discover_uncntrl_models(uncntrl_root)
    completed_poam = discover_completed_poam(result_roots, args.completion_step)

    jobs = plan_poam_jobs(args, uncntrl_models, completed_poam)
    write_manifest(Path(args.manifest), jobs)

    print("Discovered uncontrolled seeds by type:")
    for t in AGENT_TYPES:
        print(f"  {t}: {sorted(uncntrl_models.get(t, {}).keys())}")
    print(f"\nPlanned jobs: {len(jobs)}")
    for job in jobs:
        print(f"  - {job['id']}")
    print(f"\nManifest written to {args.manifest}")

    if not args.run:
        print("\nDry run only. Re-run with --run to execute the planned jobs.")
        return 0

    for job in jobs:
        run_job(job, keep_going=args.keep_going)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
