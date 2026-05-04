import argparse
import datetime
import itertools
import os
import subprocess
from copy import deepcopy

import yaml


DEFAULT_GRID = {
    "classifier_history_len": [16, 32, 64],
    "classifier_d_model": [64, 128],
    "classifier_layers": [1, 2],
    "classifier_lr": [0.0001, 0.0003],
    "classifier_dropout": [0.1],
}


def parse_csv_values(raw, cast):
    return [cast(item.strip()) for item in raw.split(",") if item.strip()]


def config_name_from_path(config_path):
    root = os.path.join("src", "config") + os.sep
    config_name = os.path.splitext(config_path)[0]
    if config_name.startswith(root):
        config_name = config_name[len(root):]
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


def write_config(base_config, combo, seed, output_dir, base_results_path):
    config = deepcopy(base_config)
    label = combo_label(combo, seed)
    config.update(combo)
    config["seed"] = seed
    config["label"] = label
    config["local_results_path"] = os.path.join(base_results_path, label)

    trained_agents = config.get("trained_agents", {})
    agent_cfg = trained_agents.get("agent_0")
    if agent_cfg and agent_cfg.get("agent_loader") == "type_conditional_loader":
        classifier_cfg = agent_cfg.setdefault("classifier", {})
        classifier_cfg["history_len"] = combo["classifier_history_len"]
        classifier_cfg["d_model"] = combo["classifier_d_model"]
        classifier_cfg["num_layers"] = combo["classifier_layers"]
        classifier_cfg["dropout"] = combo["classifier_dropout"]

    os.makedirs(output_dir, exist_ok=True)
    config_path = os.path.join(output_dir, f"{label}.yaml")
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    return label, config_path


def command_for_config(config_path, seed, args):
    cmd = [
        "python",
        "src/main.py",
        f"--seed={seed}",
        f"--env-config={args.env_config}",
        f"--config={config_name_from_path(config_path)}",
        f"--alg-config={args.alg_config}",
        "with",
    ]
    if args.cuda is not None:
        cmd.append(f"use_cuda={str(args.cuda)}")
    if args.t_max is not None:
        cmd.append(f"t_max={args.t_max}")
    if args.save_model_interval is not None:
        cmd.append(f"save_model_interval={args.save_model_interval}")
    if args.test_interval is not None:
        cmd.append(f"test_interval={args.test_interval}")
    if args.extra_override:
        cmd.extend(args.extra_override)
    if args.conda_env:
        return ["conda", "run", "-n", args.conda_env, *cmd]
    return cmd


def main():
    parser = argparse.ArgumentParser(description="Grid-search classifier-only training configs.")
    parser.add_argument("--base-config", default="src/config/open/classifier_only_train_pp.yaml")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--base-results-path", default="mpe-pp/open_train/classifier_grid")
    parser.add_argument("--env-config", default="gymma")
    parser.add_argument("--alg-config", default="mpe/poam_type_classifier")
    parser.add_argument("--conda-env", default="epy")
    parser.add_argument("--seeds", default="112358", help="Comma-separated seeds.")
    parser.add_argument("--history-lens", default=None, help="Comma-separated classifier_history_len values.")
    parser.add_argument("--d-models", default=None, help="Comma-separated classifier_d_model values.")
    parser.add_argument("--layers", default=None, help="Comma-separated classifier_layers values.")
    parser.add_argument("--lrs", default=None, help="Comma-separated classifier_lr values.")
    parser.add_argument("--dropouts", default=None, help="Comma-separated classifier_dropout values.")
    parser.add_argument("--t-max", type=int, default=None)
    parser.add_argument("--save-model-interval", type=int, default=None)
    parser.add_argument("--test-interval", type=int, default=None)
    parser.add_argument("--cuda", type=lambda x: x.lower() == "true", default=None)
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--extra-override", nargs="*", default=[], help="Additional Sacred overrides after 'with'.")
    args = parser.parse_args()

    with open(args.base_config, "r") as f:
        base_config = yaml.safe_load(f)

    timestamp = datetime.datetime.now().strftime("%m-%d-%H-%M-%S")
    output_dir = args.output_dir or f"src/config/temp/classifier_grid_{timestamp}"
    seeds = parse_csv_values(args.seeds, int)
    grid = build_grid(args)

    jobs = []
    for seed in seeds:
        for combo in iter_grid(grid):
            label, config_path = write_config(base_config, combo, seed, output_dir, args.base_results_path)
            jobs.append((label, config_path, command_for_config(config_path, seed, args)))

    if args.max_runs is not None:
        jobs = jobs[: args.max_runs]

    print(f"Generated {len(jobs)} classifier grid jobs in {output_dir}")
    for idx, (label, config_path, cmd) in enumerate(jobs, start=1):
        print(f"\n[{idx}/{len(jobs)}] {label}")
        print(f"config: {config_path}")
        print("cmd:", " ".join(cmd))
        if not args.dry_run:
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
