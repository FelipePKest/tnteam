#!/usr/bin/env bash
# Launch one collection container per teammate type and wait for all.
# Usage: bash scripts/probe_collect_parallel.sh [episodes_per_type_per_K] [teammate_seed] [tag]
#   e.g. bash scripts/probe_collect_parallel.sh 25               (in-distribution, seed 78590)
#        bash scripts/probe_collect_parallel.sh 25 112358 ood    (OOD teammates)
set -u
cd "$(dirname "$0")/.."
B=naht_results/marie_3sv5z_naht_seed78590_finished400k
N=${1:-25}
TSEED=${2:-}
TAG=${3:-}
EXTRA=""
PREFIX="3sv5z"
LOGP="collect"
if [ -n "$TSEED" ]; then EXTRA="--teammate-seed $TSEED"; fi
if [ -n "$TAG" ]; then PREFIX="3sv5z_$TAG"; LOGP="collect_$TAG"; fi
mkdir -p probe_data 3rdparty/_logs
i=0
for T in agent_ippo agent_iql agent_mappo agent_qmix agent_vdn; do
  i=$((i+1))
  docker run --rm --platform linux/amd64 --name probe_$T -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 \
    -v "$PWD":/workspace/tnteam -w /workspace/tnteam tnteam-cpu:amd64 \
    python scripts/probe_collect.py \
      --checkpoint $B/checkpoints/marie/400344 --config $B/training_config.json \
      --uncntrl-path /workspace/tnteam/uncntrl_agents \
      --teammates $T --episodes-per-type "$N" --n-uncontrolled 1 2 \
      --seed $((3000+i)) $EXTRA --out probe_data/${PREFIX}_$T.pt \
      > 3rdparty/_logs/${LOGP}_$T.log 2>&1 &
done
wait
echo "ALL DONE $(date)"
for T in agent_ippo agent_iql agent_mappo agent_qmix agent_vdn; do grep "\[collect\] saved" 3rdparty/_logs/${LOGP}_$T.log || echo "$T: NOT SAVED"; done
