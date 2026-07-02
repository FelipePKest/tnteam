#!/usr/bin/env bash
set -euo pipefail

# Usage: ./train_poam_per_uncontrolled_seed.sh [jobs_file]
# Defaults to remaining_mpe_pp_jobs.json in the repo root.

JOBS_FILE="${1:-remaining_mpe_pp_jobs.json}"

if ! [ -f "$JOBS_FILE" ]; then
  echo "Jobs file not found: $JOBS_FILE" >&2
  exit 1
fi

# Extract unique uncontrolled seeds from the jobs JSON
seeds=$(grep -oP 'uncontrolled-seed=\K[0-9]+' "$JOBS_FILE" | sort -n | uniq)

if [ -z "$(echo "$seeds" | tr -d '\n')" ]; then
  echo "No uncontrolled seeds found in $JOBS_FILE" >&2
  exit 1
fi

for seed in $seeds; do
  echo "=== Running POAM train with uncontrolled_agents_seed=$seed ==="

  # try to find matching uncontrolled-agent model paths for this seed
  find_model() {
    local alg="$1"
    # prefer models already staged under ./uncntrl_agents
    path=$(find ./uncntrl_agents -type d -name "${alg}_baseline_seed=${seed}*" -print -quit 2>/dev/null || true)
    if [ -n "$path" ]; then
      # return path relative to ./uncntrl_agents so agent_path matches expected config (base_uncntrl_path + agent_path)
      echo "$path" | sed -E 's|.*/uncntrl_agents/||'
      return
    fi

    # fallback: search the repo for model directories
    path=$(find . -type d -name "${alg}_baseline_seed=${seed}*" -print -quit 2>/dev/null || true)
    if [ -n "$path" ]; then
      # try to produce a relative path that starts after './uncntrl_agents' if possible, otherwise return the basename
      echo "$path" | sed -E 's|.*/uncntrl_agents/||; s|^\./||; s|^/||'
    else
      echo ""
    fi
  }

  ippo_path=$(find_model ippo)
  iql_path=$(find_model iql)
  mappo_path=$(find_model mappo)
  qmix_path=$(find_model qmix)
  vdn_path=$(find_model vdn)

  overrides=("with env_args.pretrained_wrapper=\"PretrainedTag\" env_args.time_limit=100 env_args.key=\"mpe:PredatorPrey-v0\"")

  # Add uncntrl_agents overrides when found
  [ -n "$ippo_path" ] && overrides+=("uncntrl_agents.agent_ippo.agent_path=\"$ippo_path\"")
  [ -n "$iql_path" ] && overrides+=("uncntrl_agents.agent_iql.agent_path=\"$iql_path\"")
  [ -n "$mappo_path" ] && overrides+=("uncntrl_agents.agent_mappo.agent_path=\"$mappo_path\"")
  [ -n "$qmix_path" ] && overrides+=("uncntrl_agents.agent_qmix.agent_path=\"$qmix_path\"")
  [ -n "$vdn_path" ] && overrides+=("uncntrl_agents.agent_vdn.agent_path=\"$vdn_path\"")

  # join overrides into a single 'with' clause
  with_clause=""
  for o in "${overrides[@]}"; do
    with_clause+="$o "
  done

  python src/main.py --config=open/open_train_pp --alg-config=mpe/poam --env-config=mpe $with_clause --seed="$seed"
done
