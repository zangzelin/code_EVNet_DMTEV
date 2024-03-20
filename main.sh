git pull

# set your wandb account, see https://wandb.ai/ 
wandb login 

# for Digits dataset 
# CUDA_VISIBLE_DEVICES=4 python EVNet_main.py --data_name Digits --num_fea_aim 64 --nu 5e-3 --epochs 900

# for Mnist dataset 
# CUDA_VISIBLE_DEVICES=5 python EVNet_main.py --data_name Mnist --num_fea_aim 600 --nu 5e-3 --epochs 900
# CUDA_VISIBLE_DEVICES=5 python EVNet_main.py --data_name Gast10k1457 --num_fea_aim 1200 --nu 5e-3 --epochs 1200 --K 3

# for Cifar dataset 
# CUDA_VISIBLE_DEVICES=5 python EVNet_main.py --data_name Cifar10 --num_fea_aim 3100 --nu 5e-3 --epochs 900 --log_interval 300

# if you dont want to use gpu, you can use CUDA_VISIBLE_DEVICES=-1
# CUDA_VISIBLE_DEVICES=-1 python EVNet_main.py --data_name Digits --num_fea_aim 64 --nu 5e-3 --epochs 90

# if you want to use your own dataset, you can use the following command:
# CUDA_VISIBLE_DEVICES=-1 python EVNet_main.py --data_name CSV --num_fea_aim 64 --nu 5e-3 --epochs 90 --data_path=data/niu/
# and the data file is data/niu/data.csv
# and the label file is data/niu/label.csv

