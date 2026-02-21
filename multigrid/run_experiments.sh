#!/bin/bash

export CUDA_VISIBLE_DEVICES=1

# Run MAPPO (shared direction info)
python main.py --algorithm IPPO --mode ppo --seed 42 --env_name MultiGrid-Meetup-Empty-15x15-v0 --wandb_project multigrid-ippo --save_path /checkpoints/mappo --backward &
MAPPO_PID=$!

# Run IPPO (local obs only)
python main.py --algorithm IPPO --mode ppo --seed 42 --env_name MultiGrid-Meetup-Empty-15x15-v0 --wandb_project multigrid-ippo --save_path /checkpoints/ippo &
IPPO_PID=$!


