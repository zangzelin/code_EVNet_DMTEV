#!/bin/bash

# Check if at least two arguments were provided (script name and agent ID are always present)
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <agent_id> <gpu_list>"
    echo "Example: $0 AGENT_ID '0,1,2'"
    exit 1
fi

# The first argument is the agent ID
AGENT_ID=$1

# The second argument is the list of GPUs
GPU_LIST=$2

# Split the GPU list into an array using comma as a delimiter
IFS=',' read -r -a GPUS <<< "$GPU_LIST"

# Launch wandb agents on specified CUDA devices
for GPU_ID in "${GPUS[@]}"
do
    CUDA_VISIBLE_DEVICES=$GPU_ID wandb agent $AGENT_ID &
    sleep 20
done

# Wait for all background processes to finish
wait
