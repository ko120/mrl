#!/bin/bash

export CUDA_VISIBLE_DEVICES=1

# Run MAPPO (shared direction info)
python main.py --algorithm MAPPO --mode ppo --seed 42 --env_name MultiGrid-Meetup-Empty-15x15-v0 --wandb_project multigrid-ippo --save_path /checkpoints/mappo --backward true &
MAPPO_PID=$!

# Run IPPO (local obs only)
python main.py --algorithm IPPO --mode ppo --seed 42 --env_name MultiGrid-Meetup-Empty-15x15-v0 --wandb_project multigrid-ippo --save_path /checkpoints/ippo --backward true &
IPPO_PID=$!

echo "MAPPO PID: $MAPPO_PID"
echo "IPPO PID: $IPPO_PID"

wait $MAPPO_PID $IPPO_PID
echo "Both experiments finished."
