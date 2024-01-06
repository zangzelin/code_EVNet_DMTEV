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
import scipy
from sklearn.decomposition import PCA
import pandas as pd
# import PoolRunner
import dataloader.cal_sigma as cal_sigma
from sklearn.metrics import pairwise_distances


class DigitsDataset(data.Dataset):
    def __init__(self, data_name="Digits", train=True, datapath="~/data"):
        self.data_name = data_name
        digit = load_digits()
        data = tensor(digit.data).float()
        label = tensor(digit.target)

        fea_name = []
        for i in range(8):
            for j in range(8):
                fea_name.append('{}_{}'.format(i, j))

        self.feature_name = np.array(fea_name)
        self.def_fea_aim = 64
        self.train_val_split(data, label, train)
        self.graphwithpca = False

    def cal_near_index(self, k=10, device="cuda", uselabel=False):
        
        os.makedirs("save_near_index", exist_ok=True)
        
        filename = "save_near_index/data_name{}K{}uselabel{}".format(
            self.data_name, k, uselabel)
        if not os.path.exists(filename):
            X_rshaped = (
                self.data.reshape(
                    (self.data.shape[0], -1)).detach().cpu().numpy()
            )
            if self.graphwithpca:
                X_rshaped = PCA(n_components=50).fit_transform(X_rshaped)
            if not uselabel:
                index = NNDescent(X_rshaped, n_jobs=-1)
                neighbors_index, neighbors_dist = index.query(X_rshaped, k=k+1)
                neighbors_index = neighbors_index[:,1:]
            else:
                dis = pairwise_distances(X_rshaped)
                M = np.repeat(self.label.reshape(1, -1), X_rshaped.shape[0], axis=0)
                dis[(M-M.T)!=0] = dis.max()+1
                neighbors_index = dis.argsort(axis=1)[:, 1:k+1]
            joblib.dump(value=neighbors_index, filename=filename)

            print("save data to ", filename)
        else:
            print("load data from ", filename)
            neighbors_index = joblib.load(filename)
        # import pdb; pdb.set_trace()
        self.neighbors_index = tensor(neighbors_index).to(device)

    def train_val_split(self, data, label, train, split_int = 4):
        n_data = data.shape[0]
        g_cpu = torch.Generator()
        g_cpu.manual_seed(0)
        rand_perm = torch.randperm(n_data, generator=g_cpu)
        split_index = n_data * split_int // 5

        if train is True:
            self.data = data[rand_perm[:split_index]]
            self.label = label[rand_perm[:split_index]]
        else:
            self.data = data[rand_perm[split_index:]]
            self.label = label[rand_perm[split_index:]]
        print("train: {} size {}".format(train, self.data.shape))

    def to_device(self, device):
        self.labelstr = [[str(int(i)) for i in self.label]]
        self.data = self.data.to(device)
        self.label = self.label.to(device)

    def __getitem__(self, index):
        return index

    def __len__(self):
        return self.data.shape[0]

    def get_dim(
        self,
    ):
        return self.data[0].shape
    
    def _CalGamma(self, v):
        a = scipy.special.gamma((v + 1) / 2)
        b = np.sqrt(v * np.pi) * scipy.special.gamma(v / 2)
        out = a / b
        return out
    
    def _CalRho(self, dist):
        dist_copy = np.copy(dist)
        row, col = np.diag_indices_from(dist_copy)
        dist_copy[row,col] = 1e16
        rho = np.min(dist_copy, axis=1)
        return rho

    def get_sigma_rho(self, X, perplexity, v_input, K=500):

        print('use kNN mehtod to find the sigma')

        X_rshaped = X.reshape((X.shape[0],-1))

        if X_rshaped.shape[1] > 100:
            X_rshaped = PCA(n_components=50).fit_transform(X_rshaped)
            print('--------------->PCA', X_rshaped.shape)

        # index = NNDescent(X_rshaped, n_jobs=-1,)
        # neighbors_index, neighbors_dist = index.query(X_rshaped, k=K )
        # neighbors_dist = np.power(neighbors_dist, 2)
        # rho = neighbors_dist[:, 1]

        dist = np.power(
            pairwise_distances(
                X.reshape((X.shape[0],-1)),
                n_jobs=-1,
                ),
                2,
                )
        rho = self._CalRho(dist)

        r = cal_sigma.PoolRunner(
            number_point = X.shape[0],
            perplexity=perplexity,
            dist=dist,
            rho=rho,
            gamma=self._CalGamma(v_input),
            v=v_input,
            pow=2)
        sigma = np.array(r.Getout())

        std_dis = np.std(rho) / np.sqrt(X.shape[1])
        print('sigma', sigma)
        print('sigma max', np.max(sigma))
        # if std_dis < 0.20 or self.same_sigma is True:
        #     # sigma[:] = sigma.mean() * 5
        #     sigma[:] = sigma.mean()
        # print('sigma', sigma)
        return rho, sigma

