git pull

# set your wandb account, see https://wandb.ai/ 
wandb login 

# for Digits dataset 
# CUDA_VISIBLE_DEVICES=4 python EVNet_main.py --data_name Digits --num_fea_aim 64 --nu 5e-3 --epochs 900

# for Mnist dataset 
# CUDA_VISIBLE_DEVICES=4 python EVNet_main.py --data_name Mnist --num_fea_aim 600 --nu 5e-3 --epochs 900

# for Cifar dataset 
CUDA_VISIBLE_DEVICES=4 python EVNet_main.py --data_name Cifar10 --num_fea_aim 600 --nu 5e-3 --epochs 900