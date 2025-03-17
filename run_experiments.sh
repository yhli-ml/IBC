#!/bin/bash

# ============================================================
# Batch Experiment Runner for IBC with nohup
# ============================================================

# Set default parameters
RESULTS_DIR="./results"
mkdir -p $RESULTS_DIR

# Color codes for better readability
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to display usage information
function show_usage {
    echo -e "${BLUE}Usage:${NC} $0 [options]"
    echo -e "${BLUE}Options:${NC}"
    echo "  -h, --help        Show this help message"
    echo "  --config FILE     Use specified config file instead of built-in experiments"
    echo "  --results-dir DIR Set custom results directory (default: ./results)"
    echo "  --dry-run         Print commands without executing them"
    echo -e "${YELLOW}Example:${NC} $0 --config my_experiments.conf"
}

# Function to check GPU availability
function check_gpu {
    local gpu_id=$1
    if ! nvidia-smi -i $gpu_id &> /dev/null; then
        echo -e "${RED}Error: GPU $gpu_id not available${NC}"
        echo "Available GPUs:"
        nvidia-smi --list-gpus
        return 1
    fi
    return 0
}

# Parse command line arguments
DRY_RUN=false
CONFIG_FILE=""
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_usage
            exit 0
            ;;
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --results-dir)
            RESULTS_DIR="$2"
            mkdir -p "$RESULTS_DIR"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            show_usage
            exit 1
            ;;
    esac
done

# Function to extract method name from script path
function get_method_name {
    local script="$1"
    # Remove path and extension
    local basename=$(basename "$script")
    local method=${basename%.*}
    echo "$method"
}

# Function to extract parameter value from argument list
function extract_param {
    local param_name="$1"
    shift
    local params=("$@")
    
    for ((i=0; i<${#params[@]}; i++)); do
        if [[ "${params[$i]}" == "$param_name" ]] && [[ $((i+1)) -lt ${#params[@]} ]]; then
            echo "${params[$i+1]}"
            return 0
        fi
    done
    
    # If parameter not found, return default values
    case "$param_name" in
        --dataset)
            echo "unknown"
            ;;
        --imb_factor)
            echo "1"
            ;;
        --noise_ratio)
            echo "0"
            ;;
        --epsilon)
            echo "0"
            ;;
        --k)
            echo "0"
            ;;
        *)
            echo "unknown"
            ;;
    esac
}

# Function to create a descriptive log filename from parameters
function create_log_path {
    local method="$1"
    shift
    local params=("$@")
    
    local dataset=$(extract_param "--dataset" "${params[@]}")
    local imb_factor=$(extract_param "--imb_factor" "${params[@]}")
    local noise_ratio=$(extract_param "--noise_ratio" "${params[@]}")
    local epsilon=$(extract_param "--epsilon" "${params[@]}")
    local k=$(extract_param "--k" "${params[@]}")
    
    # Create directory structure
    local dir_path="${RESULTS_DIR}/${method}"
    mkdir -p "$dir_path"
    
    # Create descriptive filename
    local filename="${dataset}_imb${imb_factor}_nr${noise_ratio}"
    
    # Add epsilon and k if applicable (some methods might not use them)
    if [[ "$method" != "SFA" ]]; then
        filename="${filename}_ep${epsilon}_k${k}"
    fi
    
    # Add timestamp for uniqueness
    local timestamp=$(date +"%Y%m%d_%H%M%S")
    filename="${filename}_${timestamp}.log"
    
    echo "${dir_path}/${filename}"
}

# Function to run an experiment
function run_experiment {
    local script=$1
    local name=$2
    shift 2
    local params=("$@")
    
    # Get method name from script
    local method=$(get_method_name "$script")
    
    # Create log file path based on parameters
    local log_file=$(create_log_path "$method" "${params[@]}")
    
    echo -e "${GREEN}=====================================================${NC}"
    echo -e "${GREEN}Starting experiment: ${name}${NC}"
    echo -e "${GREEN}Command:${NC} nohup python ${script} ${params[@]}"
    echo -e "${GREEN}Log:${NC} ${log_file}"
    echo -e "${GREEN}=====================================================${NC}"
    
    if [[ "$DRY_RUN" != "true" ]]; then
        # Check if GPU is specified and available
        local gpu_param=$(echo "${params[@]}" | grep -o -- "--gpuid [0-9]\+" || echo "")
        if [[ ! -z "$gpu_param" ]]; then
            local gpu_id=$(echo $gpu_param | awk '{print $2}')
            if ! check_gpu $gpu_id; then
                echo -e "${RED}Skipping experiment due to GPU issue${NC}"
                return 1
            fi
        fi
        
        # Run the experiment with nohup
        echo "Starting experiment: $name at $(date)" > "$log_file"
        echo "Command: python $script ${params[@]}" >> "$log_file"
        echo "----------------------------------------" >> "$log_file"
        
        # Execute with nohup and append to log file
        nohup python $script ${params[@]} >> "$log_file" 2>&1 &
        local pid=$!
        
        echo -e "${GREEN}Experiment started with PID: $pid${NC}"
        echo "Experiment started with PID: $pid" >> "$log_file"
        
        # Optional: you can wait for the process if you want to run experiments sequentially
        # wait $pid
    else
        echo -e "${YELLOW}[DRY RUN] Would execute:${NC} nohup python $script ${params[@]} > $log_file 2>&1 &"
    fi
    
    return 0
}

if [[ ! -z "$CONFIG_FILE" ]]; then
    if [[ ! -f "$CONFIG_FILE" ]]; then
        echo -e "${RED}Error: Config file $CONFIG_FILE not found${NC}"
        exit 1
    fi
    
    echo -e "${BLUE}Running experiments from config file: $CONFIG_FILE${NC}"
    
    # Parse and execute experiments from config file
    # Format should be: name|script|param1 param2 param3...
    while IFS= read -r line || [[ -n "$line" ]]; do
        # Skip empty lines and comments
        [[ -z "$line" || "$line" =~ ^#.* ]] && continue
        
        IFS='|' read -r name script params <<< "$line"
        # Remove any leading/trailing whitespace
        name=$(echo "$name" | xargs)
        script=$(echo "$script" | xargs)
        params=$(echo "$params" | xargs)
        
        # Convert params string to array
        IFS=' ' read -r -a param_array <<< "$params"
        
        run_experiment "$script" "$name" "${param_array[@]}"
    done < "$CONFIG_FILE"
else
    # Define experiments inline if no config file is provided
    echo -e "${BLUE}Running predefined experiments${NC}"
    
    # Example experiments - modify as needed
    
    # SFA experiments
    # add here ...
    
    # Train_cifar.py experiments (IBC)
    run_experiment "Train_cifar.py" "IBC_cifar10_exp_01" \
        --dataset cifar10 --imb_type exp --imb_factor 0.01 \
        --noise_mode imb --noise_ratio 0.2 \
        --epsilon 0.2 --k 0.5 --gpuid 0 --seed 123
    
    run_experiment "Train_cifar.py" "IBC_cifar10_exp_02" \
        --dataset cifar10 --imb_type exp --imb_factor 0.01 \
        --noise_mode imb --noise_ratio 0.5 \
        --epsilon 0.2 --k 0.5 --gpuid 0 --seed 123
        
    # WebVision experiments
    run_experiment "Train_longtailed_noisy_webvision.py" "IBC_webvision_01" \
        --num_class 50 --imb_factor 0.1 --batch_size 16 \
        --gpuid1 0 --gpuid2 1 --epsilon 0.25 --k 0.1
fi

echo -e "${GREEN}All experiments launched!${NC}"
echo -e "${GREEN}Logs available in: $RESULTS_DIR${NC}"
echo -e "${YELLOW}Note: Experiments are running in the background with nohup${NC}"
echo -e "${YELLOW}Use 'ps aux | grep python' to check running processes${NC}"