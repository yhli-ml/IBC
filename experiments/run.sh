#!/bin/bash
Method="Train_longtail_cifar"
datasets=("cifar10" "cifar100")
noise_ratios=(0.2 0.5)
imb_ratios=(0.005)
gpus=(2 2 4 7)
index=0
epsilon=0.1
k=0.25


for dataset in "${datasets[@]}"; do
    for noise_ratio in "${noise_ratios[@]}"; do
        for imb_ratio in "${imb_ratios[@]}"; do
            gpu=${gpus[$index]}
            index=$(( (index + 1) % ${#gpus[@]} ))
            file_path="./results/${Method}/${dataset}_imb${imb_ratio}_nr${noise_ratio}_ep${epsilon}_k${k}.log" 
            log_dir=$(dirname "${file_path}")
            mkdir -p "${log_dir}"
            nohup python -u ${Method}.py --epsilon ${epsilon} --k ${k} \
            --dataset ${dataset} --noise_ratio ${noise_ratio} --imb_factor ${imb_ratio} \
            --gpuid ${gpu} > ${file_path} 2>&1 &
        done
    done
done

# python Train_cifar.py --dataset cifar10 --noise_ratio 0.2 --imb_factor 0.1 --warm_up 1 --num_epoch 2 --gpuid 5