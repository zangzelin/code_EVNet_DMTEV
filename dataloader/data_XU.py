from torch.utils import data
from sklearn.datasets import load_digits
from torch import tensor
import torchvision.datasets as datasets
from pynndescent import NNDescent
import os
import joblib
import torch
import numpy as np
from PIL import Image
import scanpy as sc
import scipy, sys
from sklearn.decomposition import PCA
import pandas as pd
import scipy.sparse as sp

import pickle as pkl

from dataloader.data_sourse import DigitsDataset


class CSVDataset(DigitsDataset):
    def __init__(self, data_name="Xu_Gut", train=True, datapath="~/data"):
        # digit = load_digits()
        self.data_name = data_name
        data = pd.read_csv(datapath+'/data.csv', header=None).to_numpy().astype(np.float32)
        if os.path.exists(datapath+'/label.csv'):
            label = np.read_csv(datapath+'/label.csv')
        else:
            label = np.zeros(data.shape[0])
        
        try:
            os.remove('save_near_index/data_nameCSVK5uselabelFalse')
        except:
            pass
        
        data = tensor(data).float()
        label = tensor(label).long()
        
        self.def_fea_aim = 64
        self.data = data
        self.label = label
        # self.train_val_split(data, label, train, split_int=5)
        self.graphwithpca = False


class Xu_GutDataset(DigitsDataset):
    def __init__(self, data_name="Xu_Gut", train=True, datapath="~/data"):
        # digit = load_digits()
        self.data_name = data_name
        adata = sc.read(datapath+"/Gut/genome_kmer.h5ad")
        data = tensor(adata.obsm['X_pca'])
        label_train_str = list(adata.obs['celltype'])
        label_train_str_set = list(set(label_train_str))
        label = tensor(
            np.array([label_train_str_set.index(i) for i in label_train_str]))
        
        self.def_fea_aim = 64
        self.train_val_split(data, label, train)
        self.graphwithpca = True