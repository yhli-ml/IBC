#!/bin/bash
Method="Train_cifar_w_three_experts"
datasets=("cifar10")
noise_ratios=(0.2)
imb_ratios=(0.1)
epsilon=0.1
k=0.25


for dataset in "${datasets[@]}"; do
    for noise_ratio in "${noise_ratios[@]}"; do
        for imb_ratio in "${imb_ratios[@]}"; do
            if [ "$dataset" == "cifar10" ]; then
                data_path="/seu_nvme/datasets/open/cifar/cifar-10"
            else
                data_path="/seu_nvme/datasets/open/cifar/cifar-100"
            fi
            python -u ${Method}.py --epsilon ${epsilon} --k ${k} --dataset ${dataset} --noise_ratio ${noise_ratio} --imb_factor ${imb_ratio} --data_path ${data_path}
        done
    done
done


# python Train_cifar_w_random_select_k.py --dataset cifar10 --noise_ratio 0.2 --imb_factor 0.1 --warm_up 1 --num_epoch 2 --gpuid 5