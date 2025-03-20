#!/bin/bash

# Bash script to run long-tailed CIFAR experiments with IBC method
# Usage: bash run_longtailed_experiments.sh
running_script="Train_longtailed_cifar_semantic_smoothing"
# Create necessary directories
mkdir -p ./checkpoint
mkdir -p ./saved

# Define common parameters
common_args="--num_epochs 200 --warm_up 30 --arch resnet32"

# Define experiment configurations
# Format: "dataset imb_factor epsilon k gpuid"
declare -a experiments=(
    # CIFAR-100 experiments with different imbalance factors
    "cifar100 0.1 0.1 0.2 0"    # Imbalance ratio 1:10
    "cifar100 0.01 0.1 0.2 1"   # Imbalance ratio 1:50
)

# Run each experiment
for exp in "${experiments[@]}"
do
    # Parse experiment configuration
    read -r dataset imb_factor epsilon k gpuid <<< "$exp"
    
    # Calculate imbalance ratio for display
    imb_ratio=$(python -c "print(int(1/$imb_factor))")
    
    # Set batch size based on dataset (smaller for CIFAR-100 due to higher memory usage)
    if [[ "$dataset" == "cifar100" ]]; then
        batch_size=64
    else
        batch_size=128
    fi
    
    # Determine number of classes
    if [[ "$dataset" == "cifar100" ]]; then
        num_class=100
    else
        num_class=10
    fi
    
    # Construct log file name
    log_file="logs/${running_script}/${dataset}_imb${imb_factor}_eps${epsilon}_k${k}.log"
    
    # Create logs directory if it doesn't exist
    mkdir -p "$(dirname "$log_file")"
    
    # Print configuration
    echo "Starting experiment with the following configuration:"
    echo "Dataset: $dataset"
    echo "Imbalance factor: $imb_factor (ratio 1:$imb_ratio)"
    echo "Epsilon: $epsilon"
    echo "K: $k"
    echo "GPU ID: $gpuid"
    echo "Log file: $log_file"
    
    # Run the experiment
    echo "Running python ${running_script}.py --dataset $dataset --imb_factor $imb_factor --epsilon $epsilon --k $k --gpuid $gpuid --batch_size $batch_size --num_class $num_class $common_args"
    
    nohup python $running_script.py \
        --dataset $dataset \
        --imb_factor $imb_factor \
        --epsilon $epsilon \
        --k $k \
        --gpuid $gpuid \
        --batch_size $batch_size \
        --num_class $num_class \
        $common_args > "$log_file" 2>&1 &
    
    # Wait a bit before starting the next experiment to avoid initialization conflicts
    sleep 2
    
    echo "Experiment started in background. Check $log_file for progress."
    echo "--------------------------------------------"
done

echo "All experiments have been launched."
echo "Use 'ps aux | grep python' to see running processes."
echo "Use 'tail -f logs/*.log' to monitor all experiment logs."