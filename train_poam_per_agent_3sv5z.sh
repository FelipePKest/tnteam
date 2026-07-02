#!/bin/bash
# Script to train POAM agent separately for each uncontrolled agent type on 3s_vs_5z.
# Each training run uses only ONE agent type in the uncntrl_agents config.
#
# Usage: ./train_poam_per_agent_3sv5z.sh [agent_type]
#   - No argument: trains against all agent types sequentially
#   - agent_type: trains against a specific agent (ippo, qmix, vdn, mappo, iql)

SEED=112358
PYTHON_BIN=${PYTHON_BIN:-python3}

# List of agent types to train against
# AGENT_TYPES=("ippo" "qmix" "vdn" "mappo" "iql")
AGENT_TYPES=("ippo")


# If an argument is provided, train only against that agent type
if [ $# -eq 1 ]; then
    AGENT_TYPE_ARG=$1
    if [[ " ${AGENT_TYPES[*]} " =~ " ${AGENT_TYPE_ARG} " ]]; then
        AGENT_TYPES=("$AGENT_TYPE_ARG")
    else
        echo "Error: Invalid agent type '$AGENT_TYPE_ARG'"
        echo "Valid options: ippo, qmix, vdn, mappo, iql"
        exit 1
    fi
elif [ $# -gt 1 ]; then
    echo "Usage: ./train_poam_per_agent_3sv5z.sh [agent_type]"
    echo "Valid options: ippo, qmix, vdn, mappo, iql"
    exit 1
fi

echo "=============================================="
echo "Training POAM agent against each 3s_vs_5z agent type separately"
echo "=============================================="

for AGENT_TYPE in "${AGENT_TYPES[@]}"; do
    AGENT_TYPE_UPPER=$(printf "%s" "$AGENT_TYPE" | tr '[:lower:]' '[:upper:]')

    echo ""
    echo "=============================================="
    echo "Training POAM against ${AGENT_TYPE_UPPER} agents on 3s_vs_5z"
    echo "=============================================="

    # Use the separate SC2 3sv5z config files for each agent type.
    "${PYTHON_BIN}" src/main.py \
        --env-config=sc2 \
        --config=open/uncntrl_agents/3sv5z_${AGENT_TYPE} \
        --alg-config=sc2/poam \
        --seed=${SEED} \
        with \
        env_args.map_name=3s_vs_5z

    # Check if training succeeded
    if [ $? -eq 0 ]; then
        echo "Training against ${AGENT_TYPE_UPPER} completed successfully!"
    else
        echo "Training against ${AGENT_TYPE_UPPER} failed!"
        exit 1
    fi
done

echo ""
echo "=============================================="
echo "All 3s_vs_5z POAM training runs completed!"
echo "=============================================="
