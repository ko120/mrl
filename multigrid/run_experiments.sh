#!/bin/bash

# Run MAPPO (shared direction info)
python main.py --algorithm MAPPO --mode ppo --seed 42 --wandb_project multigrid-ippo --save_path /checkpoints/mappo &
MAPPO_PID=$!

# Run IPPO (local obs only)
python main.py --algorithm IPPO --mode ppo --seed 42 --wandb_project multigrid-ippo --save_path //checkpoints/ippo &
IPPO_PID=$!

echo "MAPPO PID: $MAPPO_PID"
echo "IPPO PID: $IPPO_PID"

wait $MAPPO_PID $IPPO_PID
echo "Both experiments finished."
