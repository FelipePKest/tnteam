#!/bin/bash
# Script to train POAM agent separately for each uncontrolled agent type
# Each training run uses only ONE agent type in the uncntrl_agents config
#
# Usage: ./train_poam_per_agent.sh [agent_type]
#   - No argument: trains against all agent types sequentially
#   - agent_type: trains against a specific agent (ippo, qmix, vdn, mappo, iql)

SEED=112358

# List of agent types to train against
AGENT_TYPES=("ippo" "qmix" "vdn" "mappo" "iql")

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
fi

echo "=============================================="
echo "Training POAM agent against each agent type separately"
echo "=============================================="

for AGENT_TYPE in "${AGENT_TYPES[@]}"; do
    AGENT_TYPE_UPPER=$(printf "%s" "$AGENT_TYPE" | tr '[:lower:]' '[:upper:]')

    echo ""
    echo "=============================================="
    echo "Training POAM against ${AGENT_TYPE_UPPER} agents"
    echo "=============================================="
    
    # Use the separate config files for each agent type
    python src/main.py \
        --env-config=mpe \
        --config=open/uncntrl_agents/pp_${AGENT_TYPE} \
        --alg-config=mpe/poam \
        --seed=${SEED} \
        with \
        env_args.key=mpe:PredatorPrey-v0 \
        env_args.pretrained_wrapper=PretrainedTag \
        env_args.time_limit=100
    
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
echo "All POAM training runs completed!"
echo "=============================================="
