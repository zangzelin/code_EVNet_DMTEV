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
    'Mnist',
    'KMnist',
    'EMnistBC',
    "arcene",
    'Gast10k1457',
    'MCAD9119',
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
Uniform_t_list = [
    0.05,
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    ]
# K_list = [3, 5, 10, ]
K_list = [5,]
num_latent_dim_list = [2]
# num_latent_dim_list = [50]
augNearRate_list = [100]
epochs_list = [9000]
# num_fea_aim_list = [200]
num_fea_aim_list = [ 64, 128, 256, 32,]
# num_fea_aim_list = [ 64]
detaalpha_list = [1.001,]
# addtopkloss_list = [0]
l2alpha_list = [20]
project_name = ['_v5_ablation_augtype_uniform_t',]
data_path_list = ['/public/home/liziqinggroup/zangzelin/data/']
data_path_list = ['/zangzelin/data']
seed_list = [0,1,2,3,4,5,6,7]
seed_list = [0]
augtype_list = [
    # 'Uniform+Normal+Bernoulli',
    # 'Uniform+Bernoulli',
    # 'Normal+Bernoulli',
    # 'Uniform+Normal',
    'Uniform', 
    # 'Bernoulli', 
    # 'Normal',
    ]


cudalist = [
    0,
    1,
    2,
    3,
    # 4,
    # 5,
    # 6,
    # 7,
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
    augtype_list,
    Uniform_t_list,
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
    'augtype',
    'Uniform_t'
    # 'addtopkloss'
    # 'l2_alpha',
]

mainFunc = "./otn_main.py"
ater = tool.AutoTrainer(
    changeList,
    paramName,
    mainFunc,
    deviceList=cudalist,
    poolNumber=4*len(cudalist),
    name="autotrain",
    waittime=1,
)
ater.Run()


