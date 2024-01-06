import functools
import os
import re
import sys
from turtle import shape

# from pathlib import Path
import plotly.express as px

# import matplotlib.pylab as plt
import numpy as np

# import pandas as pd
import pytorch_lightning as pl
import torch

# from torch._C import device
import wandb
import plotly.graph_objects as go
# from numpy.core.fromnumeric import size
from torch import nn
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import distance_matrix as dm
# from torch.nn import functional as F
from sklearn.cluster import KMeans
# from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from sklearn.cluster import SpectralClustering

from sklearn.neighbors import kneighbors_graph

from dataloader import data_base
from pytorch_lightning import LightningModule, Trainer
from torchvision import transforms
from torchmetrics import Accuracy
import Loss.dmt_loss_aug as dmt_loss_aug
import Loss.dmt_loss_old as dmt_loss_old

import eval.eval_core as ec

from aug.aug import aug_near_mix, aug_near_feautee_change, aug_randn
from sklearn.cluster import KMeans

torch.set_num_threads(2)


def gpu2np(a):
    return a.cpu().detach().numpy()

def pw_cosine_similarity(input_a, input_b):
   normalized_input_a = torch.nn.functional.normalize(input_a)  
   normalized_input_b = torch.nn.functional.normalize(input_b)
   res = torch.mm(normalized_input_a, normalized_input_b.T)
#    res *= -1 # 1-res without copy
#    res += 1
   return res

class MyLinear(nn.Linear):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # return torch.nn.functional.linear(input, self.weight, self.bias)
        return torch.nn.functional.linear(input, self.weight, self.bias)


class NN_FCBNRL(nn.Module):
    # def forward(self, input: torch.Tensor) -> torch.Tensor:
    #     return torch.nn.functional.linear(input, self.weight, self.bias)
    def __init__(self, in_dim, out_dim, use_RL=True):
        super(NN_FCBNRL, self).__init__()
        m_l = []
        m_l.append(MyLinear(in_dim, out_dim))
        m_l.append(nn.BatchNorm1d(out_dim))
        if use_RL:
            m_l.append(nn.LeakyReLU(0.1))
        self.block = nn.Sequential(*m_l)
    
    def forward(self, x):
        return self.block(x)


class LitMNIST(LightningModule):
    def __init__(
        self,
        dataname,
        **kwargs,
    ):

        super().__init__()

        # Set our init args as class attributes
        self.dataname = dataname
        # self.learning_rate = learning_rate
        self.save_hyperparameters()
        self.t = 0.1
        # self.alpha = 1e-5
        self.alpha = None
        self.stop = False
        self.detaalpha = self.hparams.detaalpha
        self.bestval = 0
        self.setup()
        self.wandb_logs = {}
        # if self.hparams.num_fea_aim < 0:
        # self.num_fea_aim = int(
        #     self.hparams.num_fea_aim*self.data_train.data.shape[1])

        # self.one_hot = self.CalOnehotMask()
        self.hparams.num_pat = min(
            self.data_train.data.shape[1], self.hparams.num_pat)
        # self.PM_root = torch.tensor(torch.ones((
        #         self.data_train.data.shape[1],
        #         self.hparams.num_pat ))/5)

        self.model_pat, self.model_b = self.InitNetworkMLP(
        # self.model_pat, self.model_b = self.InitNetworkMLP_OLD(
            self.hparams.NetworkStructure_1, 
            self.hparams.NetworkStructure_2
        )
        self.hparams.num_fea_aim = min(
            self.hparams.num_fea_aim, self.data_train.data.shape[1])

        self.Loss = dmt_loss_aug.MyLoss(
            v_input=100,
            metric=self.hparams.metric,
            augNearRate=self.hparams.augNearRate,
        )
        # self.Loss_DMT = dmt_loss_old.MyLoss(
        #     v_input=100,
        #     )
        # fea_fea_emb, fea_emb_importance = self.FindFeaEmb(
        #     ins_emb=torch.tensor(self.data_train.data.detach().cpu()),
        #     pat_val=torch.tensor(self.data_train.data.detach().cpu()),
        #     )
        # fea_fea_emb = torch.tensor(fea_fea_emb)
        # m = np.zeros(shape=(
        #     self.data_train.data.shape[1],
        #     self.hparams.num_pat
        # ))
        # kmeans = KMeans(n_clusters=self.hparams.num_pat, random_state=0).fit(fea_fea_emb)
        # for i,k in enumerate(kmeans.labels_):
        #     m[i,k]=1
        fea_fea_emb, fea_emb_importance = self.FindFeaEmb(
            ins_emb=self.data_train.data.detach().cpu(),
            pat_val=self.data_train.data.detach().cpu(),
            )
        fea_fea_emb = fea_fea_emb.detach().cpu().numpy()
        m = np.zeros(shape=(
            self.data_train.data.shape[1],
            self.hparams.num_pat
        ))
        # kmeans = KMeans(n_clusters=self.hparams.num_pat, random_state=0).fit(fea_fea_emb)
        kmeans = SpectralClustering(
            n_clusters=self.hparams.num_pat,
            assign_labels='discretize',
            random_state=0,
            eigen_solver="arpack",
            affinity="nearest_neighbors",
            ).fit(fea_fea_emb)
        
        for i,k in enumerate(kmeans.labels_):
            m[i,k]=0.5
        
        self.m =  torch.tensor(m).float().to('cuda') - 0.25 
        
        self.PM_root = nn.Parameter(
            torch.tensor(torch.ones((
                self.data_train.data.shape[1],
                self.hparams.num_pat ))/5) #+torch.tensor(m).float()
            )
        # self.m = torch.zeros_like(self.PM_root).float().to('cuda')

    # def forward_pat(self, x):
    #     maskt = self.PM_root > self.t
    #     mask0 = self.PM_root >= self.PM_root.max(dim=0)[0].reshape(1,-1)
    #     mask1 = self.PM_root >= self.PM_root.max(dim=1)[0].reshape(-1,1)

    #     self.mask = ((maskt+mask0+mask1)>0)

    #     y_list = []
    #     for i in range(self.PM_root.shape[1]):
    #         lat = x* (self.PM_root[:, i] * self.mask[:, i])
    #         y = self.model_pat[i](lat)
    #         y_list.append(y)
    #     lat1 = torch.cat(y_list, dim=1)

    #     lat3 = lat1
    #     for i, m in enumerate(self.model_b):
    #         lat3 = m(lat3)
    #     return lat1, lat1, lat3
    def forward_fea(self, x):
        def modify_grad(x, inds):
            x[inds] = 0 
            return x

    
        # if self.alpha is None:
        #     self.mask = (
        #         torch.zeros((self.data_train.data.shape[1], self.hparams.num_pat))+1
        #         ).bool()
        #     lat = x

        PatternMatrix = self.PM_root + self.m
        maskt = PatternMatrix > self.t
        mask0 = PatternMatrix >= PatternMatrix.max(dim=0)[0].reshape(1,-1)
        self.mask = ((maskt+mask0)>0)
        self.PM_root.register_hook(lambda x: modify_grad(x, ~self.mask))
        
        y_list = []
        for i in range(self.hparams.num_pat):
            lat = x * ((self.PM_root+self.m)[:, i] * self.mask[:, i])
            if self.alpha is None:
               lat = lat.detach()
            y = self.model_pat[i](lat)
            y_list.append(y)
        lat1 = torch.cat(y_list, dim=1)

        lat3 = lat1
        for i, m in enumerate(self.model_b):
            lat3 = m(lat3)
        return lat1, lat1, lat3


    def forward_fea_old(self, x):
        maskt = self.PM_root > self.t
        mask0 = self.PM_root >= self.PM_root.max(dim=0)[0].reshape(1,-1)
        # mask1 = self.PM_root >= self.PM_root.max(dim=1)[0].reshape(-1,1)

        self.mask = ((maskt+mask0)>0)

        mean_list = []
        lat1 = x
        for i, m in enumerate(self.model_pat):
            lat1 = m(lat1)
            mean_list.append(lat1)
            if i==(len(self.model_pat)-2):        
                for lat_his in mean_list:
                    lat1 += lat_his
                lat1 /= (len(self.model_pat)-2)
        
        # print(lat1.shape)

        lat3 = lat1
        for i, m in enumerate(self.model_b):
            lat3 = m(lat3)
        return lat1, lat1, lat3

    def forward(self, x):
        return self.forward_fea(x)
        # return self.forward_fea_old(x)

    # def InitFeautreModel(self, stru=[-1, 100, 2]):

    #     # stru[0] = len(self.feature_node_list)
    #     stru[0] = min(self.data_train.data.shape[0], 1000)
    #     stru[-1] = self.hparams.num_latent_dim
    #     f_model = nn.ModuleList()
    #     for i in range(len(stru) - 1):
    #         if i != len(stru) - 2:
    #             f_model.append(nn.Linear(stru[i], stru[i + 1]))
    #             # f_model.append(nn.BatchNorm1d(stru[i + 1]))
    #             f_model.append(nn.LeakyReLU())
    #         else:
    #             f_model.append(nn.Linear(stru[i], stru[i + 1]))

    #     print(f_model)
    #     return f_model

    # def training_step(self, batch, batch_idx):
    #     x, y = batch
    #     logits = self(x)
    #     loss = F.nll_loss(logits, y)
    #     return loss

    def training_step(self, batch, batch_idx):
        index = batch.to(self.device)

        # augmentation
        data1 = self.data_train.data[index]
        data2 = self.augmentation(index, data1)
        data = torch.cat([data1, data2])

        # forward
        pat, mid, lat = self(data)
        # loss
        loss_topo = self.Loss(
            input_data=mid.reshape(mid.shape[0], -1),
            latent_data=lat.reshape(lat.shape[0], -1),
            v_latent=self.hparams.nu, 
            metric='euclidean',
            # metric='cossim',
        )

        # loss_topo = self.Loss_DMT(
        #     input_data=data1,
        #     latent_data=lat,
        #     rho=self.rho[index],
        #     sigma=self.sigma[index],
        #     v_latent=self.hparams.nu,
        #     )


        self.wandb_logs = {
            "loss_topo": loss_topo,
            "lr": self.trainer.optimizers[0].param_groups[0]["lr"],
            "epoch": self.current_epoch,
            # "T": self.t_list[self.current_epoch],
        }


        loss_l2 = 0
        if (
            torch.rand(1) > 0.8
            and self.current_epoch > 600
        ):
            if self.alpha is None:
                print('if self.alpha is None:')
                
                self.alpha = loss_topo.detach().item() / (
                    self.Cal_Sparse_loss(
                        self.PM_root+self.m, self.mask, self.hparams.num_pat,
                        # self.fea_fea_dis
                        ).detach()
                    * self.hparams.l2alpha
                )
            # N_link = np.sum(gpu2np(self.mask))
            feature_use_bool = gpu2np(self.mask).sum(axis=1)>0
            N_Feature = np.sum(feature_use_bool)
            if (N_Feature > self.hparams.num_fea_aim):
                loss_l2 = self.Cal_Sparse_loss(
                    self.PM_root+self.m, self.mask, self.hparams.num_pat, 
                    # self.fea_fea_dis
                    )
                self.alpha = self.alpha * self.detaalpha
                # self.alpha_cs = self.alpha_cs * self.detaalpha
                # use_PatternMatrix = (self.PM_root * (self.PM_root > self.t)).t()            
                # loss_cs = 0
                # for i in range(self.hparams.num_pat):
                #     w = self.PM_root[:,i]
                #     pat_item = data1.t()
                #     loss_cs += (pw_cosine_similarity(pat_item, pat_item) * w).mean() * 100
                
                # wandb.log({
                #     'epoch': self.current_epoch,
                #     'loss_l2': loss_l2,
                #     # 'loss_cs': loss_cs,
                # })
                loss_topo += (loss_l2) * self.alpha
        return loss_topo 

    def validation_step(self, batch, batch_idx):
        # augmentation
        if (self.current_epoch + 1) % self.hparams.log_interval == 0:
            index = batch.to(self.device)
            data = self.data_train.data[index]

            label = np.array(self.data_train.label.cpu())[gpu2np(index)]
            pat, mid, lat = self(data)

            return (
                gpu2np(data),
                gpu2np(pat),
                gpu2np(lat),
                label,
                gpu2np(index),
            )

    def Cal_Sparse_loss(self, PatM, mask, num_pat):
        loss_l2 = 0
        for p in range(num_pat):
            c_pat_for_PM = PatM[:, p]
            mean_mask = mask.sum()/mask.shape[1]
            loss_l2 += torch.abs(
                c_pat_for_PM
                ).mean() * ( (mask[:,p].sum()/mean_mask)**2)
        return loss_l2

    def Cal_Sparse_loss_fea_fea_dis(self, PatM, mask, num_pat, fea_fea_dis):
        loss_l2 = 0
        for p in range(num_pat):
            mask_c = mask[:,p]
            w = fea_fea_dis[:,mask_c].sum(dim=1).to('cuda')
            c_pat_for_PM = PatM[:, p]
            mean_mask = mask.sum()/mask.shape[1]
            l1 = torch.abs(c_pat_for_PM) * w
            loss_l2 += l1.mean()*( (mask[:,p].sum()/mean_mask)**2)
        return loss_l2

    def validation_epoch_end(self, outputs):
        if not self.stop:
            self.log('es_monitor', self.current_epoch)
        else:
            self.log('es_monitor', 0)

        if (self.current_epoch + 1) % self.hparams.log_interval == 0:
            print("self.current_epoch", self.current_epoch)
            data = np.concatenate([data_item[0] for data_item in outputs])
            mid_old = np.concatenate([data_item[1] for data_item in outputs])
            ins_emb = np.concatenate([data_item[2] for data_item in outputs])
            label = np.concatenate([data_item[3] for data_item in outputs])
            index = np.concatenate([data_item[4] for data_item in outputs])

            num_fea_per_pat = self.hparams.num_fea_per_pat
            mid_c_list = []
            for i in range(mid_old.shape[1]//num_fea_per_pat):
                mid_c_list.append(
                    mid_old[:, i*num_fea_per_pat: (i+1)*num_fea_per_pat].mean(axis=1).reshape(-1,1)
                    )
            mid = np.concatenate(mid_c_list, axis=1)

            N_link = np.sum(gpu2np(self.mask))
            feature_use_bool = gpu2np(self.mask).sum(axis=1)>0
            N_Feature = np.sum(feature_use_bool)
            self.wandb_logs.update(
                {
                    "epoch": self.current_epoch,
                    "alpha": self.alpha,
                    "metric/#link": N_link,
                    "metric/#Feature": N_Feature,
                    "metric/PatMatrix": go.Figure(data=go.Heatmap(
                    z=gpu2np(self.PM_root+self.m)*gpu2np(self.mask))),
                }
            )
            self.wandb_logs.update(
                ec.ShowEmb(ins_emb, self.data_train.labelstr, index))
            
            # if self.current_epoch > self.hparams.epochs-300:
            if N_Feature <= self.hparams.num_fea_aim:
                # self.bestval = ec.Test_ET_CV(e, self.wandb_logs, self.bestval)
                self.stop = True
                self.wandb_logs.update(
                    self.up_fig(data, mid, mid_old, ins_emb, label, index))
            #     self.wandb_logs.update({
            #         "n_rate": e.GraphMatch(),
            #         # "n_rate_latent": e.GraphMatchLatent(),
            #         "vis/Mask": ec.showMask(self.MaskWeight, t=top_k_t),
            #         "vis/VisSelectUMAP": e.VisSelectUMAP(data, label),
            #         "vis/VisAllUMAP": e.VisAllUMAP(data, label),
            #         "selected_index": ",".join(
            #             [
            #                 str(a)
            #                 for a in self.MaskWeight.detach()
            #                 .sort()[1][(-1 * self.num_fea_aim):]
            #                 .cpu()
            #                 .numpy()
            #                 .tolist()
            #             ]
            #         ),
            #         }
            #     )
            
            if self.wandb_logs is not None:
                wandb.log(self.wandb_logs)

    def up_fig(self, data, mid, mid_old, ins_emb, label, index):
        
      
        
        # from sklearn.neighbors import KDTree
        # from scipy.sparse import csr_matrix
        wandb_logs = {}
        # fea_sort_index = gpu2np(torch.sort(self.PM_root.sum(dim=1),descending=True)[1])[:self.hparams.num_fea_aim]
        # fea_emb, fea_emb_neg = self.FindPatEmb(
        #     ins_emb=torch.tensor(ins_emb), pat_val=torch.tensor(data),)
        # pat_emb, pat_emb_neg = self.FindPatEmb(
        #     ins_emb=torch.tensor(ins_emb), pat_val=torch.tensor(mid_old),)
        # fea_emb = fea_emb[:self.hparams.num_fea_aim]
        # fea_emb_neg = fea_emb_neg[:self.hparams.num_fea_aim]
        # wandb_logs.update(ec.ShowEmbIns(ins_emb, fea_emb[fea_sort_index], fea_emb_neg[fea_sort_index], label, index, str_1='fea'))
        # wandb_logs.update(ec.ShowEmbInsColored_dence(
        #     ins_emb=ins_emb, pat_mid=data, 
        #     pat_emb=fea_emb[fea_sort_index], str_1='fea'))
        # tree = KDTree(ins_emb)
        # def FindMeanLocation(fea_emb, knn = 5):
        #     fea_emb_nn = [(data[tree.query([position], 5)[1]])[0] for position in fea_emb]
        #     fea_emb_nn_d = [tree.query([position], 5)[0] for position in fea_emb]
        #     fea_emb_nn_w = [dis/np.sum(dis) for dis in fea_emb_nn_d]
        #     fea_emb_nn = [np.dot(fea_emb_nn_w[i], fea_emb_nn[i]) for i in range(len(fea_emb_nn))]
        #     return fea_emb_nn
        
        # def LinkToPath(pre, start_index, end_index, ):
        #     # pre = pres[start_index]
        #     # if start_index != end_index:
        #     try:
        #         node = end_index
        #         path = [end_index, ]
        #         while node != start_index:
        #             node_pre = pre[node]
        #             path.append(node_pre)
        #             node = node_pre
        #         return path
        #     except:
        #         return [start_index, end_index]
        
        # def find_geodesic_curve(start, end, data_matrix, edge_num=30):
        #     expanded_data_matrix = np.concatenate(
        #         [data_matrix, np.concatenate(start), np.concatenate(end)], axis=0)
            
        #     # distance_matrix_ = dm(expanded_data_matrix, expanded_data_matrix)
        #     distance_matrix_ = kneighbors_graph(
        #         expanded_data_matrix, 
        #         n_neighbors=2, 
        #         mode='distance', 
        #         include_self=False,
        #         ).toarray()

        #     graph = csr_matrix(distance_matrix_)
        #     dis_m, pres = dijkstra(
        #         graph, directed=False, return_predecessors=True)

        #     data_list = []            
        #     for i in range(len(start)):
        #         index_start = data.shape[0]+i
        #         index_end = data.shape[0]+i+len(start)
        #         path = LinkToPath(pres[index_start], index_start, index_end)
        #         data_hd = expanded_data_matrix[path]
        #         fea_track_2d = gpu2np(
        #             self(torch.tensor(data_hd).to('cuda').float() )[-1]
        #             )
        #         data_list.append( fea_track_2d )
        #     return data_list
        
        # fea_emb_nn = FindMeanLocation(fea_emb)
        # fea_emb_neg_nn = FindMeanLocation(fea_emb_neg)
        # fea_track_list = find_geodesic_curve(
        #         fea_emb_nn, fea_emb_neg_nn, data
        #         )
        # pat_emb_nn = FindMeanLocation(pat_emb)
        # pat_emb_neg_nn = FindMeanLocation(pat_emb_neg)
        # pat_track_list = find_geodesic_curve(pat_emb_nn, pat_emb_neg_nn, data)
        
        # self.wandb_logs.update(
        #     ec.ShowEmbIns_WithTrack(
        #         ins_emb, 
        #         [fea_emb, fea_emb_neg, fea_track_list, ],
        #         [pat_emb, pat_emb_neg, pat_track_list, ],
        #         label, 
        #         mask=self.mask, index=index, str_1='fea')
        #     )
        fea_emb, fea_emb_importance = self.FindFeaEmb(
            ins_emb=torch.tensor(ins_emb), pat_val=torch.tensor(data),)
        # pat_emb, pat_emb_neg = self.FindPatEmb(
        #     ins_emb=torch.tensor(ins_emb), 
        #     pat_emb_flatten=torch.tensor(mid_old),
        #     pat_emb_mean=torch.tensor(mid)
        #     )
        pat_emb = []
        for i in range(self.mask.shape[1]):
            mask_c = self.mask[:,i].detach().cpu().numpy()
            pat_emb.append(fea_emb[mask_c].mean(axis=0).reshape(1,-1))
        pat_emb = torch.tensor(np.concatenate(pat_emb))

        wandb_logs.update(ec.ShowEmbInsFeaPat(
            ins_emb, pat_emb, pat_emb, fea_emb, fea_emb, 
            label, mask=self.mask))
        # wandb_logs.update(ec.ShowEmbInsColored_dence_fea(
        #     ins_emb=ins_emb, fea_or_pat=fea_emb_importance, 
        #     mask_feature = self.mask.sum(dim=1) >= 1,
        #     str_1='fea')
        #     )
        # wandb_logs.update(ec.ShowEmbInsColored_dence_pat(
        #     ins_emb=ins_emb, fea_or_pat=mid, str_1='pat',))
        
        wandb_logs.update(ec.ShowSankey_Zelin(self.mask))
        return wandb_logs

    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.hparams.lr, weight_decay=1e-9
        )
        self.scheduler = StepLR(
            optimizer, step_size=self.hparams.epochs // 10, gamma=0.8
        )
        return [optimizer], [self.scheduler]

    def setup(self, stage=None):

        dataset_f = getattr(data_base, self.dataname + "Dataset")
        self.data_train = dataset_f(
            data_name=self.hparams.data_name,
            train=True,
            datapath=self.hparams.data_path,
        )
        self.data_train.cal_near_index(
            device=self.device,
            k=self.hparams.K,
        )
        self.data_train.to_device("cuda")

        self.data_test = dataset_f(
            data_name=self.hparams.data_name,
            train=False,
            datapath=self.hparams.data_path,
        )
        self.data_test.to_device("cuda")

        self.dims = self.data_train.get_dim()

    def train_dataloader(self):
        return DataLoader(
            self.data_train,
            drop_last=True,
            batch_size=min(self.hparams.batch_size, self.data_train.data.shape[0]),
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.data_train,
            batch_size= min(self.hparams.batch_size, self.data_train.data.shape[0]),
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
        )

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.hparams.batch_size)

    def InitNetworkMLP(self, NetworkStructure_1, NetworkStructure_2):

        num_fea_per_pat = self.hparams.num_fea_per_pat
        struc_model_pat = [functools.reduce(lambda x, y: x * y, self.dims)] + NetworkStructure_1[1:] + [num_fea_per_pat]
        # struc_model_a = NetworkStructure_1
        struc_model_b = NetworkStructure_2 + [2]
        # struc_model_a[0] = struc_model_pat[-1]
        struc_model_b[0] = self.hparams.num_pat * num_fea_per_pat

        model_pat = nn.ModuleList()
        for fea in range(self.hparams.num_pat):
            # m_l = [nn.Dropout(p=0.01)]
            m_l = []
            for i in range(len(struc_model_pat) - 1):
                m_l.append(
                    NN_FCBNRL(
                        struc_model_pat[i], 
                        struc_model_pat[i + 1]
                    )
                )
                # if i != len(struc_model_pat) - 2:
                #     m_l.append(MyLinear(struc_model_pat[i], struc_model_pat[i + 1]))
                #     m_l.append(nn.BatchNorm1d(struc_model_pat[i + 1]))
                #     m_l.append(nn.LeakyReLU(0.1))
                # else:
                #     m_l.append(MyLinear(struc_model_pat[i], struc_model_pat[i + 1]))
                #     m_l.append(nn.BatchNorm1d(struc_model_pat[i + 1]))
                #     m_l.append(nn.LeakyReLU(0.1))
            block = nn.Sequential(*m_l)
            model_pat.append(block)

        model_b = nn.ModuleList()
        # model_b.append(nn.Flatten())
        for i in range(len(struc_model_b) - 1):
            if i != len(struc_model_b) - 2:
                model_b.append(
                    NN_FCBNRL(
                        struc_model_b[i],
                        struc_model_b[i + 1])
                )
            else:
                model_b.append(
                    NN_FCBNRL(
                        struc_model_b[i], 
                        struc_model_b[i + 1],
                        use_RL=False),
                )

        # print(model_pat)
        # print(model_a)
        print(model_b)

        return model_pat, model_b


    def InitNetworkMLP_OLD(self, NetworkStructure_1, NetworkStructure_2):

        struc_model_pat = [
            functools.reduce(lambda x, y: x * y, self.dims)
            ] + NetworkStructure_1[1:] + [self.hparams.num_pat]
        # struc_model_a = NetworkStructure_1
        struc_model_b = NetworkStructure_2 + [2]
        # struc_model_a[0] = struc_model_pat[-1]
        struc_model_b[0] = self.hparams.num_pat

        model_pat = nn.ModuleList()
        # for fea in range(self.hparams.num_pat):
            # m_l = [nn.Dropout(p=0.01)]
        # m_l = []
        for i in range(len(struc_model_pat) - 1):
            model_pat.append(
                NN_FCBNRL(struc_model_pat[i], struc_model_pat[i + 1]))
            # if i != len(struc_model_pat) - 2:
            #     m_l.append(MyLinear(struc_model_pat[i], struc_model_pat[i + 1]))
            #     m_l.append(nn.BatchNorm1d(struc_model_pat[i + 1]))
            #     m_l.append(nn.LeakyReLU(0.1))
            # else:
            #     m_l.append(MyLinear(struc_model_pat[i], struc_model_pat[i + 1]))
            #     m_l.append(nn.BatchNorm1d(struc_model_pat[i + 1]))
            #     m_l.append(nn.LeakyReLU(0.1))
        # model_pat = nn.Sequential(*m_l)
        # model_pat.append(block)

        model_b = nn.ModuleList()
        # model_b.append(nn.Flatten())
        for i in range(len(struc_model_b) - 1):
            if i != len(struc_model_b) - 2:
                model_b.append(nn.Linear(struc_model_b[i], struc_model_b[i + 1]))
                model_b.append(nn.BatchNorm1d(struc_model_b[i + 1]))
                model_b.append(nn.LeakyReLU(0.1))
            else:
                model_b.append(nn.Linear(struc_model_b[i], struc_model_b[i + 1]))
                model_b.append(nn.BatchNorm1d(struc_model_b[i + 1]))
                # model_b.append(nn.LeakyReLU(0.1))

        print(model_pat)
        print(model_b)

        return model_pat, model_b


    def CalOnehotMask(
        self,
    ):
        self.hparams.num_pat = min(self.hparams.num_pat, self.data_train.data.shape[1])
        kmeans = KMeans(n_clusters=self.hparams.num_pat, random_state=0)
        kmeans.fit(gpu2np(self.data_train.data.t()))
        kmeans_lab = torch.tensor(kmeans.labels_).long()

        N, num_class = len(kmeans_lab), self.hparams.num_pat
        one_hot = torch.zeros(N, num_class).long()
        one_hot.scatter_(
            dim=1,
            index=kmeans_lab.unsqueeze(dim=1),
            src=torch.ones(N, num_class).long(),
        )
        return one_hot.to("cuda")

    def augmentation(self, index, data1):
        data2_list = []
        if self.hparams.Uniform_t > 0:
            data_new = aug_near_mix(
                index,
                self.data_train,
                k=self.hparams.K,
                random_t=self.hparams.Uniform_t,
                device=self.device,
            )
            data2_list.append(data_new)
        if self.hparams.Bernoulli_t > 0:
            data_new = aug_near_feautee_change(
                index,
                self.data_train,
                k=self.hparams.K,
                t=self.hparams.Bernoulli_t,
                device=self.device,
            )
            data2_list.append(data_new)
        if self.hparams.Normal_t > 0:
            data_new = aug_randn(
                index,
                self.data_train,
                k=self.hparams.K,
                t=self.hparams.Normal_t,
                device=self.device,
            )
            data2_list.append(data_new)
        if (
            max(
                [
                    self.hparams.Uniform_t,
                    self.hparams.Normal_t,
                    self.hparams.Bernoulli_t,
                ]
            )
            < 0
        ):
            data_new = data1
            data2_list.append(data_new)

        if len(data2_list) == 1:
            data2 = data2_list[0]
        elif len(data2_list) == 2:
            data2 = (data2_list[0] + data2_list[1]) / 2
        elif len(data2_list) == 3:
            data2 = (data2_list[0] + data2_list[1] + data2_list[2]) / 3

        return data2

    def pdist2(self, x: torch.Tensor, y: torch.Tensor):
        # calculate the pairwise distance

        m, n = x.size(0), y.size(0)
        xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
        yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
        dist = xx + yy
        dist = torch.addmm(dist, mat1=x, mat2=y.t(), beta=1, alpha=-2)
        dist = dist.clamp(min=1e-12)
        return dist

    def Importance(self, data_c, num_fea_per_pat=16):
        # cal the importance of ins to feature
        # return imp[i,f] means degree of importance of
        # the instance i to feature f
        num_fea = data_c.shape[1]

        # data_c = data.t()
        imp = torch.zeros_like(data_c)

        pairwise_dis = self.pdist2(data_c, data_c)
        local_index = pairwise_dis.topk(
            k=10,
            largest=False,
            dim=1,
        )[1]

        for i in range(data_c.shape[0]):
            for j in range(num_fea):
                pat_index_s = j*num_fea_per_pat
                pat_index_e = j*num_fea_per_pat + num_fea_per_pat
                pat_data_for_pat_c = data_c[:, pat_index_s: pat_index_e]
                data_local = pat_data_for_pat_c[local_index[i]]
                data_global = pat_data_for_pat_c
                data_mid = data_c[i:i+1, pat_index_s: pat_index_e]
                local_score = self.pdist2(data_mid, data_local).mean()
                global_score = self.pdist2(data_mid, data_global).mean()
                import_score = global_score - local_score
                imp[i, j] = import_score
        return imp

    def Distinguishability(self, pat_data, num_fea_per_pat=1):
        # cal the importance of ins to feature
        # return imp[i,f] means degree of importance of
        # the instance i to feature f
        # print(pat_data.shape)
        num_pat = pat_data.shape[1]//num_fea_per_pat
        # data_c = data.t()
        imp = torch.zeros(size=(pat_data.shape[0], num_pat))

        K_dis = 10

        pairwise_dis = self.pdist2(pat_data, pat_data)
        local_index = pairwise_dis.topk(
            k=K_dis,
            largest=False,
            dim=1,
        )[1]

        for i in range(pat_data.shape[0]):
            for j in range(num_pat):
                pat_index_s = j*num_fea_per_pat
                pat_index_e = j*num_fea_per_pat + num_fea_per_pat
                pat_data_for_pat_c = pat_data[:, pat_index_s: pat_index_e]
                data_local = pat_data_for_pat_c[local_index[i]]
                # data_global = pat_data_for_pat_c
                data_mid = pat_data[i:i+1, pat_index_s: pat_index_e]
                local_score = torch.abs(torch.sort(
                    self.pdist2(data_mid, data_local))[1]-torch.arange(0, K_dis).float()).mean()
                # global_score = self.pdist2(data_mid, data_global).mean()
                # import_score = global_score - local_score
                imp[i, j] = local_score
        return 1-imp


    def FindFeaEmb(
        self,
        ins_emb: torch.tensor,
        pat_val: torch.tensor,
    ):

        importance = self.Distinguishability(pat_val, num_fea_per_pat=1)
        importance_t = importance.t()

        # top_k_importance_t = importance_t.topk(k=self.hparams.K_plot, dim=1)[0][
        #     :, self.hparams.K_plot - 1
        # ]
        top_k_importance_t = importance_t.max(dim=1)[1]
        # mask = torch.zeros_like(importance_t)
        # for i in range(importance_t.shape[0]):
        #     mask[i][importance_t[i] < top_k_importance_t[i]] = 1
        best_emb = ins_emb[top_k_importance_t]
        # best_emb = np.zeros((pat_val.shape[1], ins_emb.shape[1]))
        # for i in range(pat_val.shape[1]):
        #     best_emb_item = ins_emb[(mask[i] < 0.5).bool(), :].mean(dim=0)
        #     best_emb[i] = gpu2np(best_emb_item)

        # importance_t = -1 * importance.t()
        # top_k_importance_t = importance_t.topk(k=self.hparams.K_plot, dim=1)[0][
        #     :, self.hparams.K_plot - 1
        # ]
        # mask = torch.zeros_like(importance_t)
        # for i in range(importance_t.shape[0]):
        #     mask[i][importance_t[i] < top_k_importance_t[i]] = 1

        # best_emb_neg = np.zeros((pat_val.shape[1], 2))
        # for i in range(pat_val.shape[1]):
        #     best_emb_item = ins_emb[(mask[i] < 0.5).bool(), :].mean(dim=0)
        #     best_emb_neg[i] = gpu2np(best_emb_item)

        return best_emb, importance


    def FindPatEmb(
        self,
        ins_emb: torch.tensor,
        pat_emb_flatten: torch.tensor,
        pat_emb_mean: torch.tensor,
    ):
        importance = self.Distinguishability(
            pat_emb_flatten, self.hparams.num_fea_per_pat,
            )
        importance_t = importance.t()

        top_k_importance_t = importance_t.max(dim=1)[0]
        # top_k_importance_t = importance_t.topk(k=self.hparams.K_plot, dim=1)[0][
        #     :, self.hparams.K_plot - 1
        # ]
        mask = torch.zeros_like(importance_t)
        for i in range(importance_t.shape[0]):
            mask[i][importance_t[i] < top_k_importance_t[i]] = 1

        best_emb = np.zeros((pat_emb_mean.shape[1], 2))
        for i in range(pat_emb_mean.shape[1]):
            best_emb_item = ins_emb[(mask[i] < 0.5).bool(), :].mean(dim=0)
            best_emb[i] = gpu2np(best_emb_item)

        # importance_t = -1 * importance.t()
        # top_k_importance_t = importance_t.topk(k=self.hparams.K_plot, dim=1)[0][
        #     :, self.hparams.K_plot - 1
        # ]
        # mask = torch.zeros_like(importance_t)
        # for i in range(importance_t.shape[0]):
        #     mask[i][importance_t[i] < top_k_importance_t[i]] = 1

        # best_emb_neg = np.zeros((pat_emb_mean.shape[1], 2))
        # for i in range(pat_emb_mean.shape[1]):
        #     best_emb_item = ins_emb[(mask[i] < 0.5).bool(), :].mean(dim=0)
        #     best_emb_neg[i] = gpu2np(best_emb_item)

        return best_emb, best_emb

def main(args):

    pl.utilities.seed.seed_everything(1)
    info = [str(s) for s in sys.argv[1:]]
    runname = "_".join(["dmt", args.data_name, "".join(info)])

    wandb.init(
        name=runname,
        project="PatEmb" + args.__dict__["project_name"],
        entity="zangzelin",
        mode="offline" if bool(args.__dict__["offline"]) else "online",
        save_code=True,
        config=args,
    )

    model = LitMNIST(
        dataname=args.data_name,
        **args.__dict__,
    )
    early_stop = EarlyStopping(
        monitor="es_monitor", 
        patience=900, verbose=False, mode="max"
        )
    trainer = Trainer(
        gpus=1, max_epochs=args.epochs, 
        progress_bar_refresh_rate=0,
        callbacks=[early_stop],
        )
    trainer.fit(model)


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description="*** author")
    parser.add_argument("--offline", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1, metavar="S")
    parser.add_argument("--data_path", type=str, default="/zangzelin/data")
    parser.add_argument("--log_interval", type=int, default=300)
    parser.add_argument("--project_name", type=str, default="test")
    parser.add_argument(
        "--computer", type=str,
        default=os.popen("git config user.name").read()[:-1]
    )

    # data set param
    parser.add_argument(
        "--data_name",
        type=str,
        default="Mnist",
        choices=[
            "OTU",
            "Activity",
            "Gast10k1457",
            "InsEmb_Car2",
            "PBMCD2638",
            "InsEmb_Univ",
            "InsEmb_PBMC",
            'PBMC',
            "InsEmb_Colon",
            "InsEmb_Digit",
            "InsEmb_TPD_579",
            "InsEmb_TPD_579_ALL_PRO",
            "InsEmb_TPD_867",
            "Digits",
            "Mnist",
            "EMnist",
            "KMnist",
            "FMnist",
            "Coil20",
            "Coil100",
            "Smile",
            "ToyDiff",
            "SwissRoll",
            "EMnistBC",
            "EMnistBYCLASS",
            "Cifar10",
            "Colon",
            "Gast10k",
            "HCL60K50D",
            "HCL60K3037D",
            "HCL280K50D",
            "HCL280K3037D",
            "SAMUSIK",
            "MiceProtein",
            "BASEHOCK",
            "GLIOMA",
            "leukemia",
            "pixraw10P",
            "Prostatege",
            "WARPARIOP",
            "arcene",
            "MCA",
            "MCAD9119",
            "PeiHuman",
            "PeiHumanTop2",
            "E1",
        ],
    )
    parser.add_argument(
        "--n_point",
        type=int,
        default=60000000,
    )
    # model param
    parser.add_argument(
        "--metric",
        type=str,
        default="euclidean",
    )
    parser.add_argument("--detaalpha", type=float, default=1.01)
    parser.add_argument("--l2alpha", type=float, default=2)
    parser.add_argument("--nu", type=float, default=1e-2)
    parser.add_argument("--num_link_aim", type=float, default=0.2)
    parser.add_argument("--num_fea_aim", type=int, default=32*4)
    parser.add_argument("--K_plot", type=int, default=3)

    parser.add_argument("--num_fea_per_pat", type=int, default=8)  # 0.5
    parser.add_argument("--K", type=int, default=3)
    parser.add_argument("--Uniform_t", type=float, default=-1)  # 0.3
    parser.add_argument("--Bernoulli_t", type=float, default=0.4)  # 0.4
    parser.add_argument("--Normal_t", type=float, default=-1)  # 0.5

    # train param
    parser.add_argument("--NetworkStructure_1", type=list, default=[-1, 100]+[100]*5)
    parser.add_argument("--NetworkStructure_2", type=list, default=[-1, 20])
    parser.add_argument("--num_pat", type=int, default=8)
    parser.add_argument("--num_latent_dim", type=int, default=2)
    parser.add_argument("--augNearRate", type=float, default=100)
    parser.add_argument("--batch_size",type=int,default=1000,);
    parser.add_argument("--epochs", type=int, default=50000)
    parser.add_argument("--lr", type=float, default=1e-2, metavar="LR")

    args = pl.Trainer.add_argparse_args(parser)
    args = args.parse_args()

    main(args)