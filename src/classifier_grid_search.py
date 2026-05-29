#!/usr/bin/env python3
import argparse
import datetime
import itertools
import os
import subprocess
from copy import deepcopy
from pathlib import Path

import yaml


DEFAULT_AGENT_TYPES = ("ippo", "qmix", "vdn", "mappo", "iql")

DEFAULT_GRID = {
    "classifier_history_len": [16, 32, 64],
    "classifier_d_model": [64, 128],
    "classifier_layers": [1, 2],
    "classifier_lr": [0.0003],
    "classifier_dropout": [0.1],
}


def parse_csv_values(raw, cast):
    return [cast(item.strip()) for item in raw.split(",") if item.strip()]


def config_name_from_path(config_path):
    root = os.path.join("src", "config") + os.sep
    config_name = os.path.splitext(config_path)[0]
    if config_name.startswith(root):
        config_name = config_name[len(root) :]
    return config_name


def build_grid(args):
    grid = deepcopy(DEFAULT_GRID)
    if args.history_lens:
        grid["classifier_history_len"] = parse_csv_values(args.history_lens, int)
    if args.d_models:
        grid["classifier_d_model"] = parse_csv_values(args.d_models, int)
    if args.layers:
        grid["classifier_layers"] = parse_csv_values(args.layers, int)
    if args.lrs:
        grid["classifier_lr"] = parse_csv_values(args.lrs, float)
    if args.dropouts:
        grid["classifier_dropout"] = parse_csv_values(args.dropouts, float)
    return grid


def iter_grid(grid):
    keys = list(grid.keys())
    for values in itertools.product(*(grid[key] for key in keys)):
        yield dict(zip(keys, values))


def combo_label(combo, seed):
    return (
        "classifier_grid"
        f"_h{combo['classifier_history_len']}"
        f"_d{combo['classifier_d_model']}"
        f"_l{combo['classifier_layers']}"
        f"_lr{combo['classifier_lr']:.0e}"
        f"_do{combo['classifier_dropout']}"
        f"_seed{seed}"
    ).replace(".", "p").replace("-", "m")


def _has_loadable_poam_checkpoint(model_dir):
    best_dir = model_dir / "best"
    sacred_config = Path(str(model_dir).replace("/models/", "/sacred/"), "1", "config.json")
    return (
        best_dir.is_dir()
        and (best_dir / "agent.th").is_file()
        and (best_dir / "encoder.th").is_file()
        and sacred_config.is_file()
    )


def discover_3sv5z_teammate_types(models_root, agent_types):
    """Find one POAM expert model per uncontrolled type under 3sv5z/open_train."""
    root = Path(models_root)
    teammate_types = []
    missing = []

    for agent_type in agent_types:
        models_dir = root / f"poam-vs-{agent_type}" / "models"
        candidates = [
            path
            for path in models_dir.glob("poam_*")
            if path.is_dir() and _has_loadable_poam_checkpoint(path)
        ]
        if not candidates:
            missing.append(str(models_dir))
            continue

        selected = max(candidates, key=lambda path: path.stat().st_mtime)
        teammate_types.append(
            {
                "name": agent_type,
                "agent_loader": "poam_eval_agent_loader",
                "agent_path": selected.as_posix(),
                "load_step": "best",
                "test_mode": True,
            }
        )

    if missing:
        raise FileNotFoundError(
            "Could not find loadable POAM model runs for: " + ", ".join(missing)
        )

    return teammate_types


def apply_classifier_settings(config, combo):
    config.update(combo)
    config.setdefault("classifier_nhead", 4)
    config.setdefault("classifier_ff", 256)
    config.setdefault("classifier_weight_decay", 0.0)


def configure_type_conditional_agent(config, combo, teammate_types):
    classifier_cfg = {
        "history_len": combo["classifier_history_len"],
        "d_model": combo["classifier_d_model"],
        "nhead": config.get("classifier_nhead", 4),
        "num_layers": combo["classifier_layers"],
        "ff": config.get("classifier_ff", 256),
        "dropout": combo["classifier_dropout"],
    }
    config["trained_agents"] = {
        "agent_0": {
            "agent_loader": "type_conditional_loader",
            "classifier": classifier_cfg,
            "teammate_types": deepcopy(teammate_types),
        }
    }


def write_config(base_config, combo, seed, output_dir, base_results_path, teammate_types, n_uncontrolled):
    config = deepcopy(base_config)
    label = combo_label(combo, seed)

    apply_classifier_settings(config, combo)
    configure_type_conditional_agent(config, combo, teammate_types)

    config["seed"] = seed
    config["label"] = label
    config["local_results_path"] = os.path.join(base_results_path, label)
    config["learner"] = "classifier_learner"
    config["agent"] = "rnn_poam"
    if n_uncontrolled is not None:
        config["n_uncontrolled"] = n_uncontrolled
    config.setdefault("ed_hidden_dim", 64)
    config.setdefault("embed_dim", 64)
    config.setdefault("obs_agent_id", True)
    config.setdefault("obs_state", False)
    config.setdefault("obs_individual_obs", False)
    config.setdefault("obs_team_composition", False)
    config.setdefault("batch_size", 256)
    config.setdefault("buffer_size", 256)

    os.makedirs(output_dir, exist_ok=True)
    config_path = os.path.join(output_dir, f"{label}.yaml")
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    return label, config_path


def command_for_config(config_path, seed, args):
    cmd = [
        args.python_bin,
        "src/main.py",
        f"--seed={seed}",
        f"--env-config={args.env_config}",
        f"--config={config_name_from_path(config_path)}",
        f"--alg-config={args.alg_config}",
    ]
    overrides = []
    if args.map_name:
        overrides.append(f"env_args.map_name={args.map_name}")
    if args.cuda is not None:
        overrides.append(f"use_cuda={str(args.cuda)}")
    if args.t_max is not None:
        overrides.append(f"t_max={args.t_max}")
    if args.save_model_interval is not None:
        overrides.append(f"save_model_interval={args.save_model_interval}")
    if args.test_interval is not None:
        overrides.append(f"test_interval={args.test_interval}")
    if args.extra_override:
        overrides.extend(args.extra_override)
    if overrides:
        cmd.append("with")
        cmd.extend(overrides)
    if args.conda_env:
        return ["conda", "run", "-n", args.conda_env, *cmd]
    return cmd


def main():
    parser = argparse.ArgumentParser(
        description="Grid-search uncontrolled-agent classifier training on 3s_vs_5z."
    )
    parser.add_argument("--base-config", default="src/config/open/open_train_3sv5z.yaml")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--base-results-path", default="3sv5z/open_train/classifier_grid")
    parser.add_argument(
        "--models-root",
        default="naht_results/3sv5z/open_train",
        help="Root containing poam-vs-*/models directories.",
    )
    parser.add_argument(
        "--agent-types",
        default=",".join(DEFAULT_AGENT_TYPES),
        help="Comma-separated uncontrolled types to classify.",
    )
    parser.add_argument("--env-config", default="sc2")
    parser.add_argument("--alg-config", default="mpe/poam_type_classifier")
    parser.add_argument("--map-name", default="3s_vs_5z")
    parser.add_argument("--python-bin", default="python3")
    parser.add_argument("--conda-env", default=None)
    parser.add_argument("--seeds", default="112358", help="Comma-separated seeds.")
    parser.add_argument("--history-lens", default=None)
    parser.add_argument("--d-models", default=None)
    parser.add_argument("--layers", default=None)
    parser.add_argument("--lrs", default=None)
    parser.add_argument("--dropouts", default=None)
    parser.add_argument("--t-max", type=int, default=None)
    parser.add_argument(
        "--n-uncontrolled",
        type=int,
        default=None,
        help="Override n_uncontrolled. Defaults to the base config value, usually null for sampled POAM training.",
    )
    parser.add_argument("--save-model-interval", type=int, default=None)
    parser.add_argument("--test-interval", type=int, default=None)
    parser.add_argument("--cuda", type=lambda x: x.lower() == "true", default=None)
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--extra-override",
        nargs="*",
        default=[],
        help="Additional Sacred overrides appended after 'with'.",
    )
    args = parser.parse_args()

    with open(args.base_config, "r") as f:
        base_config = yaml.safe_load(f)

    timestamp = datetime.datetime.now().strftime("%m-%d-%H-%M-%S")
    output_dir = args.output_dir or f"src/config/temp/classifier_grid_3sv5z_{timestamp}"
    seeds = parse_csv_values(args.seeds, int)
    grid = build_grid(args)
    agent_types = parse_csv_values(args.agent_types, str)
    teammate_types = discover_3sv5z_teammate_types(args.models_root, agent_types)

    jobs = []
    for seed in seeds:
        for combo in iter_grid(grid):
            label, config_path = write_config(
                base_config,
                combo,
                seed,
                output_dir,
                args.base_results_path,
                teammate_types,
                args.n_uncontrolled,
            )
            jobs.append((label, config_path, command_for_config(config_path, seed, args)))

    if args.max_runs is not None:
        jobs = jobs[: args.max_runs]

    print(f"Discovered {len(teammate_types)} teammate expert models:")
    for teammate in teammate_types:
        print(f"  {teammate['name']}: {teammate['agent_path']}")

    print(f"\nGenerated {len(jobs)} classifier grid jobs in {output_dir}")
    for idx, (label, config_path, cmd) in enumerate(jobs, start=1):
        print(f"\n[{idx}/{len(jobs)}] {label}")
        print(f"config: {config_path}")
        print("cmd:", " ".join(cmd))
        if not args.dry_run:
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
