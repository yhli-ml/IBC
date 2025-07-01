import time
import subprocess

total_seconds=30

for noise_ratio in [0.2, 0.5]:
    for imb_ratio in [0.1, 0.02, 0.01]:
        command=f"python Train_cifar_w_three_experts.py --epsilon 0.1 --k 0.25 --dataset cifar100 --noise_ratio {noise_ratio} --imb_factor {imb_ratio} --data_path /seu_nvme/datasets/open/cifar/cifar-100"
        subprocess.run(command, shell=True)
        time.sleep(total_seconds)