#!/bin/bash
Method="CE"
datasets=("cifar100")
noise_ratios=(0.2)
imb_ratios=(0.01)
gpus=(2 2 4 7)
index=0

time=$(date +"%Y-%m-%d_%H:%M:%S")

for dataset in "${datasets[@]}"; do
    for noise_ratio in "${noise_ratios[@]}"; do
        for imb_ratio in "${imb_ratios[@]}"; do
            gpu=${gpus[$index]}
            index=$(( (index + 1) % ${#gpus[@]} ))
            file_path="./results/${Method}/${dataset}_imb${imb_ratio}_nr${noise_ratio}.log" 
            log_dir=$(dirname "${file_path}")
            mkdir -p "${log_dir}"
            nohup python -u ${Method}.py \
            --dataset ${dataset} --noise_ratio ${noise_ratio} --imb_factor ${imb_ratio} \
            --gpuid ${gpu} --warm_up 200 > ${file_path} 2>&1 &
        done
    done
done


# python SFA.py --dataset cifar100 --noise_ratio 0.2 --imb_factor 0.1 --warm_up 1 --num_epoch 2 --gpuid 7