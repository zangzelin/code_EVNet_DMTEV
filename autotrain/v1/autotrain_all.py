import os
from multiprocessing import Process, Manager
import numpy as np
import signal
import time
from itertools import product
import subprocess
import socket

# parameter analysis for SAGloss

import tool

n_point_list = [600000]
data_name_list = [
    'MiceProtsein',
    'Coil20',
    'Activity', 
    # 'Mnist', 
    # 'KMnist', 
    # 'EMnistBC',
    'GLIOMA', 
    'leukemia', 
    'pixraw10P', 
    'Prostatege', 
    'arcene', 
    'Gast10k1457',
    'PBMCD2638',
    'MCAD9119',
    'Colon',
    'HCL60K3037D',
    ]
perplexity_list = [20,]
# lr_list = [1e-2, 5e-3] 
lr_list = [1e-2] 
# batch_size_list = [300, 3000]
batch_size_list = [1000]
vs_list = [1e-3,]
ve_list = [-1]
method_list = ['dmt']
# K_list = [3, 5, 10, ]
K_list = [5]
num_latent_dim_list = [50, 2]
# num_latent_dim_list = [50]
augNearRate_list = [100]
epochs_list = [9000]
num_fea_aim_list = [16, 32, 64, 128, 256, 512]
detaalpha_list = [1.001]
# addtopkloss_list = [0]
l2alpha_list = [20]
project_name = ['_v5_ablation_number_feature',]
data_path_list = ['/public/home/liziqinggroup/zangzelin/data/']
data_path_list = ['/root/data/']
seed_list = [0,1,2,3,4,5,6,7]
seed_list = [0]


cudalist = [
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
]

changeList = [
    n_point_list,
    data_name_list,
    perplexity_list,
    batch_size_list,
    lr_list,
    vs_list,
    ve_list,
    method_list,
    K_list,
    num_latent_dim_list,
    augNearRate_list,
    detaalpha_list,
    l2alpha_list,
    epochs_list,
    project_name,
    data_path_list,
    num_fea_aim_list,
    seed_list,
    # addtopkloss_list,
    # l2_alpha_list,
    ]

paramName = [
    'n_point',
    'data_name',
    'perplexity',
    'batch_size',
    'lr',
    'vs',
    've',
    'method',
    'K',
    'num_latent_dim',
    'augNearRate',
    'detaalpha',
    'l2alpha',
    'epochs',
    'project_name',
    'data_path',
    'num_fea_aim',
    'seed',
    # 'addtopkloss'
    # 'l2_alpha',
]

mainFunc = "./main_new.py"
ater = tool.AutoTrainer(
    changeList,
    paramName,
    mainFunc,
    deviceList=cudalist,
    poolNumber=1*len(cudalist),
    name="autotrain",
    waittime=1,
)
ater.Run()


