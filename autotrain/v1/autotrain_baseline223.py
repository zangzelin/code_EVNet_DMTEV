import os
from multiprocessing import Process, Manager
import numpy as np
import signal
import time
from itertools import product
import subprocess

# parameter analysis for SAGloss

import tool

n_point_list = [6000000]
# data_name_list = ['PBMC', 'Digits', 'Mnist', 'Coil20',]

data_name_list = [
    'MiceProtein', 
    'Coil20', 'leukemia', 'Activity', 'GLIOMA',
    'pixraw10P', 'Mnist', 'FMnist', 'Prostatege', 'arcene',
    'KMnist', 'HCL280K3037D', 'MCAD9119'
    ]
# data_name_list = ['Gast10k1457', 'MCA', 'HCL280K3037D']
data_name_list = [
    # 'MiceProtein', 
    'Coil20', 
    'Activity', 
    'Mnist', 
    'KMnist', 
    'GLIOMA', 
    'leukemia', 
    'pixraw10P', 
    'Prostatege', 
    'arcene', 
    'Gast10k1457',
    'EMnistBC',
    'MCAD9119',
    'HCL280K3037D', 
    ] 
# data_name_list = ['arcene']
method_list = [
    'UDFS', 
    # 'MCFS', 
    'NDFS', 
    # 'lap_score', 
    # 'aefs', 
    # 'pfa', 
    # 'CAE',
    # 'FAE'
    ]
# method_list = ['aefs', 'UDFS', 'MCFS', 'NDFS', 'lap_score', 'pfa', 'CAE']
# method_list = ['aefs', 'MCFS', 'lap_score', 'pfa']
# 'UDFS' 'NDFS' 
# method_list = ['CAE']
num_fea_list = [16, 50, 64, 128, 200, 600]
# seed_list = [1,2,3,4,5,6,7,8,9,10]
seed_list = [1]

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
method_list,
num_fea_list,
seed_list,
    ]

paramName = [
    'n_point',
    'data_name',
    'method',
    'num_fea',
    'seed'
]

mainFunc = "./baseline.py"
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


