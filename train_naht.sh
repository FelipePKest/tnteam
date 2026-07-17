#!/bin/bash
expname="3sv5z-naht-poam"
logdir="~/naht_results/3sv5z/poam"
dt=$(date '+%d-%m-%Y-%H-%M-%S')

# Create log directory if it does not exist
mkdir -p "$logdir"

# train POAM on starcraft
python src/main.py --env-config=sc2 --config=open/open_train_3sv5z --alg-config=sc2/poam with env_args.map_name=3s_vs_5z --seed=1285842