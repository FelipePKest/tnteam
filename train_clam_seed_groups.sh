#!/usr/bin/env bash
set -euo pipefail

# Train one CLAM agent per complete Predator-Prey teammate-seed group.
# The 112358 group is intentionally excluded because it has already been trained.
# Runs are sequential: the next seed starts only after the previous run succeeds.
#
# Usage:
#   ./train_clam_seed_groups.sh
#   DRY_RUN=1 ./train_clam_seed_groups.sh
#   PYTHON_BIN=/path/to/python ./train_clam_seed_groups.sh

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

SEEDS=(38410 78590 93718 1285842)
ALGORITHMS=(ippo iql mappo qmix vdn)
BASE_UNCNTRL_PATH="${BASE_UNCNTRL_PATH:-$REPO_ROOT/uncntrl_agents}"
DRY_RUN="${DRY_RUN:-0}"

if [[ -n "${PYTHON_BIN:-}" ]]; then
    : # Use the caller-provided interpreter.
elif [[ -x "$HOME/.conda/envs/mbnaht/bin/python" ]]; then
    PYTHON_BIN="$HOME/.conda/envs/mbnaht/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
else
    echo "No Python interpreter found. Activate mbnaht or set PYTHON_BIN." >&2
    exit 1
fi

declare -A MODEL_PATHS

echo "Preflighting teammate checkpoints under: $BASE_UNCNTRL_PATH"
for seed in "${SEEDS[@]}"; do
    for algorithm in "${ALGORITHMS[@]}"; do
        model_dir="$BASE_UNCNTRL_PATH/mpe-pp/$algorithm/models"
        mapfile -t matches < <(
            find "$model_dir" -mindepth 1 -maxdepth 1 -type d \
                -name "${algorithm}_baseline_seed=${seed}_*" -print 2>/dev/null | sort
        )

        if (( ${#matches[@]} != 1 )); then
            echo "Expected exactly one $algorithm checkpoint for seed $seed; found ${#matches[@]}." >&2
            if (( ${#matches[@]} > 0 )); then
                printf '  %s\n' "${matches[@]}" >&2
            fi
            exit 1
        fi

        # Agent paths are relative to base_uncntrl_path in user_info.yaml.
        MODEL_PATHS["$seed:$algorithm"]="${matches[0]#"$BASE_UNCNTRL_PATH/"}"
    done
done

echo "Preflight passed for all ${#SEEDS[@]} seed groups."
echo "Python: $PYTHON_BIN"

for seed in "${SEEDS[@]}"; do
    echo
    echo "=== Starting CLAM training for teammate seed group $seed ==="

    command=(
        "$PYTHON_BIN" src/main.py
        --env-config=mpe
        --config=open/open_train_pp
        --alg-config=mpe/clam
        with
        'env_args.pretrained_wrapper="PretrainedTag"'
        env_args.time_limit=100
        'env_args.key="mpe:PredatorPrey-v0"'
        "uncntrl_agents.agent_ippo.agent_path=\"${MODEL_PATHS["$seed:ippo"]}\""
        "uncntrl_agents.agent_iql.agent_path=\"${MODEL_PATHS["$seed:iql"]}\""
        "uncntrl_agents.agent_mappo.agent_path=\"${MODEL_PATHS["$seed:mappo"]}\""
        "uncntrl_agents.agent_qmix.agent_path=\"${MODEL_PATHS["$seed:qmix"]}\""
        "uncntrl_agents.agent_vdn.agent_path=\"${MODEL_PATHS["$seed:vdn"]}\""
        --seed="$seed"
    )

    if [[ "$DRY_RUN" == "1" ]]; then
        printf 'DRY RUN: '
        printf '%q ' "${command[@]}"
        printf '\n'
    else
        "${command[@]}"
        echo "=== Finished CLAM training for teammate seed group $seed ==="
    fi
done

echo
if [[ "$DRY_RUN" == "1" ]]; then
    echo "Dry run complete; no training was started."
else
    echo "All four CLAM seed-group training runs completed."
fi
