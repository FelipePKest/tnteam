#!/usr/bin/env bash
set -euo pipefail

# Screen the most promising CLAM type-supervision improvements.
#
# Safe default: print commands without starting training.
#
# Examples:
#   ./train_clam_type_supervised_sweep.sh
#   DRY_RUN=0 SEEDS="112358" EXPERIMENTS="baseline split_temp" \
#     ./train_clam_type_supervised_sweep.sh
#   DRY_RUN=0 SEEDS="112358 38410 78590" T_MAX=10050000 \
#     ./train_clam_type_supervised_sweep.sh
#
# Available experiments:
#   baseline       Reproduce the successful 1:1 combined objective.
#   split_temp     Sharpen instance and type discrimination temperatures.
#   stronger_type  Double the type-supervised coefficient.
#   prefix_mix     Mix deployment-aligned prefixes with random crops.
#   embed32        Prefix mixture with a wider actor-facing representation.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

DRY_RUN="${DRY_RUN:-1}"
T_MAX="${T_MAX:-10050000}"
SEEDS_TEXT="${SEEDS:-112358 38410 78590}"
EXPERIMENTS_TEXT="${EXPERIMENTS:-baseline split_temp stronger_type prefix_mix embed32}"
BASE_UNCNTRL_PATH="${BASE_UNCNTRL_PATH:-$REPO_ROOT/uncntrl_agents}"

read -r -a SEED_LIST <<< "$SEEDS_TEXT"
read -r -a EXPERIMENT_LIST <<< "$EXPERIMENTS_TEXT"
ALGORITHMS=(ippo iql mappo qmix vdn)

if [[ "$DRY_RUN" != "0" && "$DRY_RUN" != "1" ]]; then
    echo "DRY_RUN must be 0 or 1." >&2
    exit 1
fi
if ! [[ "$T_MAX" =~ ^[1-9][0-9]*$ ]]; then
    echo "T_MAX must be a positive integer." >&2
    exit 1
fi

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
if [[ "$DRY_RUN" == "0" ]] && ! "$PYTHON_BIN" -c \
    'import einops, sacred, torch, yaml' >/dev/null 2>&1; then
    echo "Python at $PYTHON_BIN is missing a required training dependency." >&2
    echo "Activate the training environment or set PYTHON_BIN explicitly." >&2
    exit 1
fi

model_path() {
    local seed="$1"
    local algorithm="$2"
    local model_root="$BASE_UNCNTRL_PATH/mpe-pp/$algorithm/models"
    local matches=()
    local candidate

    if [[ ! -d "$model_root" ]]; then
        echo "Missing teammate model directory: $model_root" >&2
        return 1
    fi
    while IFS= read -r candidate; do
        matches+=("$candidate")
    done < <(
        find "$model_root" -mindepth 1 -maxdepth 1 -type d \
            -name "${algorithm}_baseline_seed=${seed}_*" -print | sort
    )

    if (( ${#matches[@]} != 1 )); then
        echo "Expected exactly one $algorithm checkpoint for seed $seed; found ${#matches[@]}." >&2
        if (( ${#matches[@]} > 0 )); then
            printf '  %s\n' "${matches[@]}" >&2
        fi
        return 1
    fi
    printf '%s\n' "${matches[0]#"$BASE_UNCNTRL_PATH/"}"
}

append_experiment_overrides() {
    local experiment="$1"
    case "$experiment" in
        baseline)
            EXPERIMENT_OVERRIDES=(
                clam_instance_temperature=0.5
                clam_supervised_temperature=0.5
                clam_instance_coef=1.0
                clam_supervised_coef=1.0
                clam_prefix_crop_probability=0.0
            )
            ;;
        split_temp)
            EXPERIMENT_OVERRIDES=(
                clam_instance_temperature=0.2
                clam_supervised_temperature=0.2
                clam_instance_coef=1.0
                clam_supervised_coef=1.0
                clam_prefix_crop_probability=0.0
            )
            ;;
        stronger_type)
            EXPERIMENT_OVERRIDES=(
                clam_instance_temperature=0.2
                clam_supervised_temperature=0.2
                clam_instance_coef=1.0
                clam_supervised_coef=2.0
                clam_prefix_crop_probability=0.0
            )
            ;;
        prefix_mix)
            EXPERIMENT_OVERRIDES=(
                clam_instance_temperature=0.2
                clam_supervised_temperature=0.2
                clam_instance_coef=1.0
                clam_supervised_coef=1.0
                clam_prefix_crop_probability=0.5
                clam_min_crop=16
                clam_max_crop=64
                clam_mask_ratio=0.2
            )
            ;;
        embed32)
            EXPERIMENT_OVERRIDES=(
                clam_instance_temperature=0.2
                clam_supervised_temperature=0.2
                clam_instance_coef=1.0
                clam_supervised_coef=1.0
                clam_prefix_crop_probability=0.5
                clam_min_crop=16
                clam_max_crop=64
                clam_mask_ratio=0.2
                embed_dim=32
            )
            ;;
        *)
            echo "Unknown experiment: $experiment" >&2
            echo "Choose from: baseline split_temp stronger_type prefix_mix embed32" >&2
            return 1
            ;;
    esac
}

echo "Preflighting teammate checkpoints under: $BASE_UNCNTRL_PATH"
for seed in "${SEED_LIST[@]}"; do
    if ! [[ "$seed" =~ ^[0-9]+$ ]]; then
        echo "Invalid seed: $seed" >&2
        exit 1
    fi
    for algorithm in "${ALGORITHMS[@]}"; do
        model_path "$seed" "$algorithm" >/dev/null
    done
done
for experiment in "${EXPERIMENT_LIST[@]}"; do
    append_experiment_overrides "$experiment"
done

echo "Preflight passed."
echo "Python: $PYTHON_BIN"
echo "Seeds: ${SEED_LIST[*]}"
echo "Experiments: ${EXPERIMENT_LIST[*]}"
echo "Training horizon: $T_MAX"

for experiment in "${EXPERIMENT_LIST[@]}"; do
    append_experiment_overrides "$experiment"
    for seed in "${SEED_LIST[@]}"; do
        ippo_path="$(model_path "$seed" ippo)"
        iql_path="$(model_path "$seed" iql)"
        mappo_path="$(model_path "$seed" mappo)"
        qmix_path="$(model_path "$seed" qmix)"
        vdn_path="$(model_path "$seed" vdn)"
        run_label="type_${experiment}_screen"

        command=(
            "$PYTHON_BIN" src/main.py
            --env-config=mpe
            --config=open/open_train_pp
            --alg-config=mpe/clam_type_supervised
            "--seed=$seed"
            "--label=$run_label"
            with
            'env_args.pretrained_wrapper="PretrainedTag"'
            env_args.time_limit=100
            'env_args.key="mpe:PredatorPrey-v0"'
            "t_max=$T_MAX"
            "uncntrl_agents.agent_ippo.agent_path=\"$ippo_path\""
            "uncntrl_agents.agent_iql.agent_path=\"$iql_path\""
            "uncntrl_agents.agent_mappo.agent_path=\"$mappo_path\""
            "uncntrl_agents.agent_qmix.agent_path=\"$qmix_path\""
            "uncntrl_agents.agent_vdn.agent_path=\"$vdn_path\""
            "${EXPERIMENT_OVERRIDES[@]}"
        )

        echo
        echo "=== $experiment | seed $seed ==="
        if [[ "$DRY_RUN" == "1" ]]; then
            printf 'DRY RUN: '
            printf '%q ' "${command[@]}"
            printf '\n'
        else
            "${command[@]}"
            echo "=== Finished $experiment | seed $seed ==="
        fi
    done
done

echo
if [[ "$DRY_RUN" == "1" ]]; then
    echo "Dry run complete; no training was started. Set DRY_RUN=0 to run it."
else
    echo "All requested CLAM type-supervision experiments completed."
fi
