import subprocess
import multiprocessing as mp


def run_training(noise_ratio, imb_ratio):
    command = f"python Train_cifar_w_three_experts.py --epsilon 0.1 --k 0.25 --dataset cifar10 --noise_ratio {noise_ratio} --imb_factor {imb_ratio} --data_path /seu_nvme/datasets/open/cifar/cifar-10"
    subprocess.run(command, shell=True)

if __name__ == "__main__":
    processes = []

    for noise_ratio in [0.2, 0.5]:
        for imb_ratio in [0.1, 0.02, 0.01]:
            p = mp.Process(target=run_training, args=(noise_ratio, imb_ratio))
            processes.append(p)
            p.start()

    for p in processes:
        p.join()
