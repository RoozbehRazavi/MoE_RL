#!/bin/bash

# Threshold for GPU utilization (e.g., 10%)
threshold=10
# Duration to check low utilization (e.g., 5 minutes)
duration_threshold=300
# Current duration of low utilization
low_duration=0
# Check interval in seconds
interval=1000

# Function to start your training process
start_training() {
    echo "Starting new training process..."
    # Command to start another training process
    python main.py --model_c model3 --exp_name pacman_mixture_embedding_dqn_16_new_arch --channel_wise_input false
}

while true; do
    # Get GPU utilization
    utilization=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits)

    # Check if utilization is below the threshold
    if [[ $utilization -lt $threshold ]]; then
        # Increment the low utilization duration
        low_duration=$((low_duration + interval))
    else
        # Reset the low utilization duration
        low_duration=0
    fi

    # Check if the low utilization duration exceeds the threshold
    if [[ $low_duration -ge $duration_threshold ]]; then
        start_training
        # Reset the low utilization duration to avoid multiple starts
        low_duration=0
    fi

    # Wait for the next interval
    sleep $interval
done
