#!/usr/bin/env bash
set -euo pipefail

# Train a separate CLAM policy against each uncontrolled agent type.
#
# Usage:
#   ./train_clam_per_agent.sh              # train all agent types
#   ./train_clam_per_agent.sh ippo         # train one agent type
#   SEED=123 ./train_clam_per_agent.sh iql # override the training seed
#   DRY_RUN=1 ./train_clam_per_agent.sh    # print commands only

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

SEED="${SEED:-112358}"
DRY_RUN="${DRY_RUN:-0}"
AGENT_TYPES=(qmix vdn mappo iql)

if [[ $# -gt 1 ]]; then
    echo "Usage: $0 [ippo|qmix|vdn|mappo|iql]" >&2
    exit 1
fi

if [[ $# -eq 1 ]]; then
    requested_type="$1"
    valid_type=0
    for agent_type in "${AGENT_TYPES[@]}"; do
        if [[ "$requested_type" == "$agent_type" ]]; then
            valid_type=1
            break
        fi
    done

    if [[ "$valid_type" -ne 1 ]]; then
        echo "Invalid agent type: $requested_type" >&2
        echo "Valid options: ${AGENT_TYPES[*]}" >&2
        exit 1
    fi
    AGENT_TYPES=("$requested_type")
fi

if [[ -n "${PYTHON_BIN:-}" ]]; then
    : # Use the caller-provided interpreter.
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python)"
else
    echo "No Python interpreter found; set PYTHON_BIN explicitly." >&2
    exit 1
fi

for agent_type in "${AGENT_TYPES[@]}"; do
    config="src/config/open/uncntrl_agents/pp_${agent_type}.yaml"
    if [[ ! -f "$config" ]]; then
        echo "Missing uncontrolled-agent config: $config" >&2
        exit 1
    fi

    echo "=== Training CLAM against ${agent_type^^} agents (seed $SEED) ==="
    command=(
        "$PYTHON_BIN" src/main.py
        --env-config=mpe
        "--config=open/uncntrl_agents/pp_${agent_type}"
        --alg-config=mpe/clam
        "--seed=$SEED"
        with
        'env_args.key="mpe:PredatorPrey-v0"'
        'env_args.pretrained_wrapper="PretrainedTag"'
        env_args.time_limit=100
        "local_results_path=mpe-pp/open_train/clam-vs-${agent_type}"
    )

    if [[ "$DRY_RUN" == "1" ]]; then
        printf 'DRY RUN: '
        printf '%q ' "${command[@]}"
        printf '\n'
    else
        "${command[@]}"
        echo "=== Finished CLAM training against ${agent_type^^} agents ==="
    fi
done

if [[ "$DRY_RUN" == "1" ]]; then
    echo "Dry run complete; no training was started."
else
    echo "All requested CLAM training runs completed."
fi
