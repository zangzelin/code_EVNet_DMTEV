import functools
import math
import os
import sys

import numpy as np
import plotly.graph_objects as go
import pytorch_lightning as pl
import torch
from munkres import Munkres
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from sklearn.cluster import SpectralClustering
# from sklearn.cluster import KMeans
from torch import nn
from torch.nn import init
from torch.nn.parameter import Parameter
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import shap
import scipy

# import lime
# import lime.lime_tabular

import uuid

# from transformers import RagRetriever

import cf_expalain
import eval.eval_core as ec
import eval.eval_core_base as ecb
import Loss.dmt_loss_aug2 as dmt_loss_aug
# import Loss.dmt_loss_aug as dmt_loss_aug1
import wandb
from aug.aug import aug_near_feautee_change, aug_near_mix, aug_randn
from dataloader import data_base

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


# class MyLinearBMM(nn.Linear):
#     def __init__(
#         self,
#         in_features: int,
#         out_features: int,
#         channel: int,
#         bias: bool = True,
#         rescon: bool = True,
#     ) -> None:
#         super(nn.Linear, self).__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         self.rescon = rescon
#         self.channel = channel
#         self.weight = Parameter(
#             torch.Tensor(
#                 channel,
#                 in_features,
#                 out_features,
#             )
#         )
#         if bias:
#             self.bias = Parameter(torch.Tensor(channel, out_features))
#         else:
#             self.register_parameter("bias", None)
#         self.reset_parameters()

#     def reset_parameters(self) -> None:
#         init.kaiming_uniform_(self.weight, a=math.sqrt(5))
#         if self.bias is not None:
#             fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
#             bound = 1 / math.sqrt(fan_in)
#             init.uniform_(self.bias, -bound, bound)

#     def forward(self, input: torch.Tensor) -> torch.Tensor:
#         if len(input.shape) != len(self.weight.shape):
#             input = input.reshape((*input.shape, 1)).expand(
#                 (
#                     *input.shape,
#                     self.channel,
#                 )
#             )
#         out = torch.bmm(
#             input.permute(2, 0, 1), self.weight.to(input.device)
#         ) + self.bias[:, None, :].to(input.device)
#         out = out.permute(1, 2, 0)
#         if self.rescon:
#             out += input
#         return out

#     def extra_repr(self) -> str:
#         return "in_features={}, out_features={}, bias={}, rescon={}, channel{}".format(
#             self.in_features,
#             self.out_features,
#             self.bias is not None,
#             self.rescon,
#             self.channel,
#         )


# class NN_FCBNRL_BMM(nn.Module):
#     # def forward(self, input: torch.Tensor) -> torch.Tensor:
#     #     return torch.nn.functional.linear(input, self.weight, self.bias)
#     def __init__(self, in_dim, out_dim, channel=8, use_RL=True, rescon=False):
#         super(NN_FCBNRL_BMM, self).__init__()
#         m_l = []
#         m_l.append(MyLinearBMM(
#             in_dim, out_dim, channel=channel, rescon=rescon))
#         if use_RL:
#             m_l.append(nn.BatchNorm1d(out_dim))
#             m_l.append(nn.LeakyReLU(0.1))
#         self.block = nn.Sequential(*m_l)

#     def forward(self, x):
#         return self.block(x)


class NN_FCBNRL_MM(nn.Module):
    # def forward(self, input: torch.Tensor) -> torch.Tensor:
    #     return torch.nn.functional.linear(input, self.weight, self.bias)
    def __init__(self, in_dim, out_dim, channel=8, use_RL=True):
        super(NN_FCBNRL_MM, self).__init__()
        m_l = []
        m_l.append(
            nn.Linear(
                in_dim,
                out_dim,
            )
        )
        m_l.append(nn.BatchNorm1d(out_dim))
        if use_RL:
            m_l.append(nn.LeakyReLU(0.1))
        self.block = nn.Sequential(*m_l)

    def forward(self, x):
        return self.block(x)


class LitPatNN(LightningModule):
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
        self.aim_cluster = None
        self.importance = None
        self.setup()
        self.wandb_logs = {}
        self.mse = torch.nn.CrossEntropyLoss()
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
            self.hparams.NetworkStructure_2,
        )
        self.hparams.num_fea_aim = min(
            self.hparams.num_fea_aim, self.data_train.data.shape[1]
        )

        self.Loss = dmt_loss_aug.MyLoss(
            v_input=100,
            metric=self.hparams.metric,
            augNearRate=self.hparams.augNearRate,
        )

        self.classification_layer =nn.Sequential(
         nn.Linear(2, 100),
         nn.LeakyReLU(),
         nn.Linear(100,100),
         nn.LeakyReLU(),
         nn.Linear(100, gpu2np(self.data_train.label.max())+1)
        )

        # self.m = self.updata_m(init=True)
        self.PM_root = nn.Parameter(
            torch.tensor(
                torch.ones((self.data_train.data.shape[1])) / 5
            )
        )

    def forward_fea(self, x):

        lat = torch.zeros(x.shape).to(x.device)
        self.mask = self.PM_root > 0.1
        # for i in range(self.hparams.num_pat):
        if self.alpha is not None:
            lat[:, :] = x * (
                (self.PM_root) * self.mask)
        else:
            lat[:, :] = x * (
                (self.PM_root) * self.mask
                ).detach()
        lat1 = self.model_pat(lat)
        lat3 = lat1
        for i, m in enumerate(self.model_b):
            lat3 = m(lat3)
        return lat1, lat1, lat3

    def forward_exp(self, x):

        # lat = torch.zeros(x.shape).to(x.device)
        # self.mask = self.PM_root > 0.1
        # for i in range(self.hparams.num_pat):
        # if self.alpha is not None:
        x_ = x * ((self.PM_root) * self.mask)
        # else:
        #     lat[:, :] = x * (
        #         (self.PM_root) * self.mask
        #         ).detach()
        lat1 = self.model_pat(x_)
        lat3 = lat1
        for i, m in enumerate(self.model_b):
            lat3 = m(lat3)
        return lat1, lat1, lat3

    def forward_classi(self, lat):
        return torch.softmax(self.classification_layer(lat), dim=1)

    def forward(self, x):
        return self.forward_fea(x)
    
    def forward_save(self, x):
        return self.forward_exp(x)

    def predict(self, x):
        x = torch.tensor(x.to_numpy())
        return gpu2np(self.forward_simi(x))

    def predict_(self, x):
        x_b = torch.zeros((x.shape[0], self.mask.shape[0]))
        x_b[:,self.mask>0] = torch.tensor(x).float()
        sim = gpu2np(self.forward_simi(x_b)).reshape(-1,1)
        out = np.concatenate([sim, 1-sim], axis=1)
        return out

    def predict_lime_one_epoch(self, x):
        x_b = torch.zeros((x.shape[0], self.mask.shape[0])).to(self.mask.device)
        x_b[:,self.mask>0] = torch.tensor(x).float().to(self.mask.device)
        lat = gpu2np(self(x_b)[2])
        predict_r = []
        for i in range(self.cluster_centers.shape[0]):
            predict_r.append(np.exp(
                -1* np.linalg.norm(lat-self.cluster_centers[i], axis=1, keepdims=True)))
        predict = np.concatenate(predict_r, axis=1)

        return scipy.special.softmax(predict, axis=1)
    
    def predict_lime(self, x):
        out = []
        if x.shape[0] > 1000:
            for e in range(int((x.shape[0]-0.5)//1000)+1):
                x_c = x[e*1000:min((e+1)*1000, x.shape[0])]
                out_c = self.predict_lime_one_epoch(x_c)
                out.append(out_c)
            return np.concatenate(out)
        else:
            return self.predict_lime_one_epoch(x)

    def predict_lime_g(self, x):

        lat = self.forward_exp(x)[2]
        predict_r = []
        for i in range(self.cluster_centers.shape[0]):
            dis = torch.norm(lat-self.cluster_centers[i], dim=1, keepdim=True)
            if self.aim_cluster and i == self.aim_cluster:
                dis = dis * self.cluster_rescale[i]
            predict_r.append(torch.exp(-1* dis))
        predict = torch.cat(predict_r, dim=1)

        return torch.softmax(predict, dim=1)

    def predict_lime_g_for_numpy(self, x):
        predict = self.predict_lime_g(torch.tensor(x).float().to(self.device))
        return gpu2np(predict)

    def forward_simi(self, x):
        x = torch.tensor(x).to(self.mask.device)
        out = self.forward_fea(x)[2]
        dis = torch.norm(out - torch.tensor(self.cf_aim).to(x.device), dim=1)
        return torch.exp(-1 * dis).reshape(-1)

    def match_cluster(self, m1, predict_labels, num_pat):

        label = np.argmax(m1, axis=1)

        l1 = list(set(range(num_pat)))
        l2 = list(set(range(num_pat)))
        numclass1, numclass2 = len(l1), len(l1)

        cost = np.zeros((numclass1, numclass2), dtype=int)
        for i, c1 in enumerate(l1):
            mps = [i1 for i1, e1 in enumerate(label) if e1 == c1]
            for j, c2 in enumerate(l1):
                mps_d = [i1 for i1 in mps if predict_labels[i1] == c2]
                cost[i][j] = len(mps_d)

        # match two clustering results by Munkres algorithm
        m = Munkres()
        cost = cost.__neg__().tolist()
        indexes = m.compute(cost)

        # get the match results
        new_predict = np.zeros(len(predict_labels))
        for i, c in enumerate(l1):
            # correponding label in l2:
            c2 = l2[indexes[i][1]]
            # ai is the index with label==c2 in the pred_label list
            ai = [ind for ind, elm in enumerate(predict_labels) if elm == c2]
            new_predict[ai] = c

        return new_predict.astype(np.int32)

    def training_step(self, batch, batch_idx):
        index = batch.to(self.device)
        # augmentation
        data1 = self.data_train.data[index]
        data2 = self.augmentation(index, data1)
        data = torch.cat([data1, data2])

        label1 = self.data_train.label[index]
        label2 = self.data_train.label[index]
        # label2 = self.augmentation(index, label1)
        label = torch.cat([label1, label2])

        # forward
        pat, mid, lat = self(data)

        # loss
        loss_topo = self.Loss(
            input_data=mid.reshape(mid.shape[0], -1),
            latent_data=lat.reshape(lat.shape[0], -1),
            v_latent=self.hparams.nu,
            metric="euclidean",
            # metric='cossim',
        )

        # loss_topo1 = self.Loss1(
        #     input_data=mid.reshape(mid.shape[0], -1),
        #     latent_data=lat.reshape(lat.shape[0], -1),
        #     v_latent=self.hparams.nu,
        #     metric="euclidean",
        #     # metric='cossim',
        # )


        pre_dict = self.forward_classi(lat.detach())
        loss_mse = self.mse(pre_dict, label,)

        # loss_topo = self.Loss_DMT(
        #     input_data=data1,
        #     latent_data=lat,
        #     rho=self.rho[index],
        #     sigma=self.sigma[index],
        #     v_latent=self.hparams.nu,
        #     )

        self.wandb_logs = {
            "loss_mse": loss_mse,
            "loss_topo": loss_topo,
            "lr": self.trainer.optimizers[0].param_groups[0]["lr"],
            "epoch": self.current_epoch,
            # "T": self.t_list[self.current_epoch],
        }

        loss_l2 = 0
        if torch.rand(1) > 0.8 and self.current_epoch >= 300:
            if self.alpha is None:
                print("--->")
                self.alpha = loss_topo.detach().item() / (
                    self.Cal_Sparse_loss(
                        self.PM_root,
                        self.mask,
                        self.hparams.num_pat,
                    ).detach()
                    * self.hparams.l2alpha
                )

            N_Feature = np.sum(gpu2np(self.mask) > 0)
            if N_Feature > self.hparams.num_fea_aim:
                loss_l2 = self.Cal_Sparse_loss(
                    self.PM_root,
                    self.mask,
                    self.hparams.num_pat,
                )
                self.alpha = self.alpha * self.detaalpha
                loss_topo += (loss_l2) * self.alpha
        return loss_topo + loss_mse/1000

    def validation_step(self, batch, batch_idx):
        # augmentation
        if (self.current_epoch + 1) % self.hparams.log_interval == 0:
            index = batch.to(self.device)
            data = self.data_train.data[index]
            pat, mid, lat = self(data)

            return (
                gpu2np(data),
                gpu2np(pat),
                gpu2np(lat),
                np.array(self.data_train.label.cpu())[gpu2np(index)],
                gpu2np(index),
            )

    def Cal_Sparse_loss(self, PatM, mask, num_pat):
        loss_l2 = torch.abs(PatM).mean()
        return loss_l2

    def validation_epoch_end(self, outputs):
        if not self.stop:
            self.log("es_monitor", self.current_epoch)
        else:
            self.log("es_monitor", 0)

        if (self.current_epoch + 1) % self.hparams.log_interval == 0:
            print("self.current_epoch", self.current_epoch)
            data = np.concatenate([data_item[0] for data_item in outputs])
            mid_old = np.concatenate([data_item[1] for data_item in outputs])
            ins_emb = np.concatenate([data_item[2] for data_item in outputs])
            label = np.concatenate([data_item[3] for data_item in outputs])
            index = np.concatenate([data_item[4] for data_item in outputs])

            self.data = data
            self.mid_old = mid_old
            self.ins_emb = ins_emb
            self.label = label
            self.index = index

            N_link = np.sum(gpu2np(self.mask))
            feature_use_bool = gpu2np(self.mask) > 0
            N_Feature = np.sum(feature_use_bool)
            
            if self.alpha is not None and N_Feature <= self.hparams.num_fea_aim :
                ecb_e_train = ecb.Eval(input=data,latent=ins_emb,label=label,k=10)
                data_test = self.data_test.data
                label_test = self.data_test.label
                _, _, lat_test = self(data_test)
                ecb_e_test = ecb.Eval(input=gpu2np(data_test),
                    latent=gpu2np(lat_test), label=gpu2np(label_test), k=10)

                # SVC = ecb_e.E_Classifacation_SVC()

                self.wandb_logs.update(
                    {
                        "epoch": self.current_epoch,
                        "alpha": self.alpha,
                        "metric/#link": N_link,
                        "metric/#Feature": N_Feature,
                        # "metric/mask": go.Figure(data=go.Heatmap(
                            # z=gpu2np(self.mask).astype(np.float32))),
                        # "metric/PM_root": go.Figure(
                            # data=go.Heatmap(z=gpu2np(self.PM_root))
                        # ),
                        # "metric/m": go.Figure(data=go.Heatmap(z=gpu2np(self.m))),
                        # "metric/PatMatrix+mask": go.Figure(
                            # data=go.Heatmap(
                                # z=gpu2np(self.PM_root) * gpu2np(self.mask)
                            # )
                        # ),
                        # "metric/PatMatrix": go.Figure(
                            # data=go.Heatmap(z=gpu2np(self.PM_root))
                        # ),
                        'SVC_train': ecb_e_train.E_Classifacation_SVC(), #SVC= ecb_e.E_Classifacation_SVC(),
                        'SVC_test': ecb_e_test.E_Classifacation_SVC(), #SVC= ecb_e.E_Classifacation_SVC(),
                        # 'main/scatter': px.scatter
                    }
                )

                ec.ShowEmb(ins_emb, self.data_train.labelstr, index)

                # self.log('SVC', SVC_value)
                # if self.current_epoch > self.hparams.epochs-300:
                # self.wandb_logs.update(
                #     self.up_fig(data, mid_old, mid_old, ins_emb, label, index)
                # )
            else:
                self.wandb_logs.update(
                    {
                        "ShowEmb": go.Figure(data=ec.ShowEmb_return_fig(
                            ins_emb, self.data_train.labelstr, index)),
                        "epoch": self.current_epoch,
                        "alpha": self.alpha,
                        "metric/#link": N_link,
                        "metric/#Feature": N_Feature,
                    })
            # self.wandb_logs.update(ec.ShowEmb(
            #         ins_emb, self.data_train.labelstr, index))

            if N_Feature <= self.hparams.num_fea_aim:
                self.stop = True
            else:
                self.stop = False

            if self.wandb_logs is not None:
                wandb.log(self.wandb_logs)
        
        else:
            self.log('SVC', 0)
    
    def local_lime(self, data:np.ndarray, label:np.ndarray, ins_emb:np.ndarray):


        
        #     feature_name_all = np.array(
        #         ['f_{}_{}'.format(i//8, i%8) for i in range(data.shape[1])])        
        # else:
        feature_name_all = np.array(
            ['f_{}'.format(i) for i in range(data.shape[1])])
        data_after_mask = data[:, gpu2np(self.mask)]
        data_name_mask = feature_name_all[gpu2np(self.mask)]

        # explainer = lime.lime_tabular.LimeTabularExplainer(
        #     data_after_mask, 
        #     training_labels=label,
        #     feature_names=data_name_mask, 
        #     # class_names=iris.target_names, 
        #     discretize_continuous=True,
        #     verbose=True, 
        #     # mode='regression',
        #     )
        explainer = shap.KernelExplainer(
            self.predict_lime, 
            data_after_mask, 
            link="logit")

        
        l_list = np.where(label==6)[0]
        fig_dict = {}
        for i in range(5):
            data_exp = data_after_mask[l_list[i]]
            fig_img, fig_cla_predict = self.lime_exp_one(data_exp, ins_emb, explainer, data_after_mask, data)
            fig_dict['a_{}'.format(i)] = fig_img
        fig_dict['b'] = fig_cla_predict

        wandb.log(fig_dict)


    def lime_exp_one(self, data_exp, ins_emb, explainer, data_after_mask, data):
        if self.hparams.data_name == 'InsEmb_Digit':
            pix = 8
        if self.hparams.data_name == 'Mnist':
            pix = 28
        

        # shap_values = explainer.shap_values(X_test, nsamples=100)
        # shap_values = explainer.shap_values(data_exp, nsamples=100)

        # exp = explainer.explain_instance(
        #     data_exp, 
        #     self.predict_lime, 
        #     top_labels=10,
        #     num_features=15,
        #     )
        exp_list = exp.as_list()

        # scatter plot
        fig_cla_predict = go.Figure()
        fig_cla_predict.add_trace(
            go.Scatter(
                x=ins_emb[:,0],
                y=ins_emb[:,1],
                mode='markers',
                marker=dict(
                    size=[5] * ins_emb.shape[0],
                    color=np.argmax(
                    self.predict_lime(data_after_mask), axis=1),
                )
            )
        )

        # heatmap plot
        data_exp_show = np.zeros(data.shape[1])
        data_exp_show[gpu2np(self.mask)] = data_exp
        data_exp_show[~gpu2np(self.mask)] = None
        fig_img = go.Figure()
        fig_img.add_trace(
            go.Heatmap(z=data_exp_show.reshape(pix, pix)[::-1])
        )
        important_feature_list = []
        important_feature_n_p = []
        for exp_c in exp_list:
            important_feature_list.append(
                float(exp_c[0].split('_')[1].split(' ')[0]))
            important_feature_n_p.append(exp_c[1])

        important_feature_list = np.array(important_feature_list)
        important_feature_list_x = important_feature_list%pix
        important_feature_list_y = (pix-1)-important_feature_list//pix
        important_feature_n_p = np.array(important_feature_n_p)

        fig_img.add_trace(
            go.Scatter(
                x=important_feature_list_x,
                y=important_feature_list_y,
                mode="markers",
                text=['f'+str(int(a))+'_'+str(b)[:6] for (a, b) in zip(important_feature_list, important_feature_n_p)],
                marker=dict(
                    size= 15* np.abs(important_feature_n_p)/np.abs(important_feature_n_p).min() ,
                    color= important_feature_n_p>0,
                )
                )
        )
        return fig_img, fig_cla_predict

    def up_mainfig(self, data, ins_emb, label, index, mask):

        # fig = make_subplots(rows=1, cols=2)
        # import plotly.graph_objects as go

        if self.data_train.data.shape[0]>20000:
            self.rand_index = torch.randperm(
                self.data_train.data.shape[0])[:10000]
        else:
            self.rand_index = torch.tensor([i for i in range(self.data_train.data.shape[0])])

        fig = make_subplots(
            rows=2, cols=3,
            column_widths=[0.4, 0.3, 0.3],
            row_heights=[0.6, 0.4],
            vertical_spacing=0.05,
            horizontal_spacing=0.05,
            specs=[
                [{"type": "xy"}, {"type": "xy"}, {"type": "xy"}],
                [{"type": "xy"}, {"type": "sankey"}, {"type": "xy"}]
           ],
            subplot_titles=(
                "sample Embedding", "sample-feature-pattern Embedding", "Switching Embedding",
                "Global Explainability", "Local Explainability", "Switching Explainability"
                )
        )

        fig.add_traces(
            [ec.ShowEmb_return_fig(ins_emb, self.data_train.labelstr, index)],
            rows=[1], cols=[1],
        )
        
        if self.hparams.data_name == 'InsEmb_Digit':
            pix = 8
        elif self.hparams.data_name == 'Mnist':
            pix = 28
        else:
            pix = 0
        
        fig_list1, mask_fea_pat, shap_values_fea_pat = ec.show_local_expl(
            data, ins_emb, self, 
            n_feverycluset=15, n_clusters=10, pix=pix,
            num_s_shap=2,
            )
        fig.add_traces(fig_list1, rows=[1]*len(fig_list1), cols=[2]*len(fig_list1),)
        
        fig.add_traces(
            [ec.ShowSankey_Zelin_return_fig(mask_fea_pat, shap_values_fea_pat)],
            rows=[2], cols=[2],
            )

        img_from, cf = cf_expalain.CF_case_study_abili_Proto(
            model=self,
            data_use=self.data,
            datashow_lat=self.ins_emb,
            # cf_index_from=[5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
            cf_index_from=[5, 10],
            cf_index_to=0,
            num_data=10000,
        )
        cf_emb, img_from_emb, fig_31, fig_32 = ec.load_cf_explain(
            model=self,
            ins_emb=self.ins_emb,
            cf=cf,
            img_from=img_from
        )
        fig.add_traces(fig_31, rows=[1]*len(fig_31), cols=[3]*len(fig_31),)
        fig.add_traces(fig_32, rows=[2]*len(fig_32), cols=[3]*len(fig_32),)

        fig.add_annotation(
            x=cf_emb[0, 0],  # arrows' head
            y=cf_emb[0, 1],  # arrows' head
            # ax=data_show_line[0, 0],  # arrows' tail
            ax=img_from_emb[0, 0],  # arrows' tail
            ay=img_from_emb[0, 1],  # arrows' tail
            xref='x3',
            yref='y3',
            axref='x3',
            ayref='y3',
            text='',  # if you want only the arrow
            showarrow=True,
            width=2,
            arrowhead=5,
            arrowsize=5,
            arrowwidth=1,
            arrowcolor='red',
            row=1, col=3,
            )
        fig.add_annotation(
            x=cf_emb[1, 0],  # arrows' head
            y=cf_emb[1, 1],  # arrows' head
            # ax=data_show_line[0, 0],  # arrows' tail
            ax=img_from_emb[1, 0],  # arrows' tail
            ay=img_from_emb[1, 1],  # arrows' tail
            xref='x3',
            yref='y3',
            axref='x3',
            ayref='y3',
            text='',  # if you want only the arrow
            showarrow=True,
            width=2,
            arrowhead=5,
            arrowsize=5,
            arrowwidth=1,
            arrowcolor='red',
            row=1, col=3,
            )


        fig.add_trace(
            ec.Show_global_importance_Zelin_return_fig(
                gpu2np(self.PM_root) * gpu2np(self.mask)),
            row=2, col=1,
        )

        fig.update_layout(
            height=1200, width=2000,
            showlegend=False,
            title_text="ENV Result of {}".format(self.hparams.data_name)
            )
        
        fig.write_html('save_html/{}_{}_{}.html'.format(
            self.hparams.data_name, self.current_epoch, str(uuid.uuid1())))

        return fig


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
            uselabel=bool(self.hparams.uselabel),
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
            batch_size=min(self.hparams.batch_size,
                           self.data_train.data.shape[0]),
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.data_train,
            batch_size=min(self.hparams.batch_size,
                           self.data_train.data.shape[0]),
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
        )

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.hparams.batch_size)

    def InitNetworkMLP(self, NetworkStructure_1, NetworkStructure_2):

        num_fea_per_pat = self.hparams.num_fea_per_pat
        struc_model_pat = (
            [functools.reduce(lambda x, y: x * y, self.dims)]
            + NetworkStructure_1[1:]
            + [num_fea_per_pat]
        )
        # struc_model_a = NetworkStructure_1
        struc_model_b = NetworkStructure_2 + [2]
        # struc_model_a[0] = struc_model_pat[-1]
        struc_model_b[0] = num_fea_per_pat

        m_l = []
        for i in range(len(struc_model_pat) - 1):
            m_l.append(
                NN_FCBNRL_MM(
                    struc_model_pat[i],
                    struc_model_pat[i + 1],
                    # channel=self.hparams.num_pat,
                    # rescon=(i != 0) and (i != len(struc_model_pat) - 2),
                )
            )
        model_pat = nn.Sequential(*m_l)

        model_b = nn.ModuleList()
        for i in range(len(struc_model_b) - 1):
            if i != len(struc_model_b) - 2:
                model_b.append(NN_FCBNRL_MM(
                    struc_model_b[i], struc_model_b[i+1]))
            else:
                model_b.append(
                    NN_FCBNRL_MM(struc_model_b[i],
                    struc_model_b[i + 1], use_RL=False),
                )

        print(model_pat)
        print(model_b)

        return model_pat, model_b

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

    def Distinguishability(self, pat_data, num_fea_per_pat=1):
        # cal the importance of ins to feature
        # return imp[i,f] means degree of importance of
        # the instance i to feature f
        # print(pat_data.shape)
        num_pat = pat_data.shape[1] // num_fea_per_pat
        # data_c = data.t()

        K_dis = self.hparams.K_plot

        pairwise_dis = self.pdist2(pat_data, pat_data)
        local_index = pairwise_dis.topk(
            k=K_dis,
            largest=False,
            dim=1,
        )[1]
        pat_data_np = gpu2np(pat_data)
        pairwise_dis_np = gpu2np(pairwise_dis)
        local_index_np = gpu2np(local_index)

        # imp = torch.zeros(size=(pat_data_np.shape[0], num_pat))
        # for i in range(pat_data_np.shape[0]):
        #     for j in range(pat_data_np.shape[1]):
        #         c_index_local = local_index_np[i][1:K_dis]
        #         dis_local  = pat_data_np[:, j][c_index_local]
        #         if np.sum(dis_local) < 1e-3:
        #             dis_local += np.random.randn(
        #                 dis_local.shape[0])*1e-4
        #         dis_global = pairwise_dis_np[i][ c_index_local ]
        #         imp[i,j] = np.corrcoef( dis_local, dis_global)[0, 1]

        imp2 = np.zeros(shape=(pat_data_np.shape[0], num_pat))
        for i in range(pat_data_np.shape[0]):
            c_index_local = local_index_np[i][1:K_dis]
            dis_all_fea = pairwise_dis_np[i][c_index_local]
            c_pat_data_np = pat_data_np[c_index_local]
            dis_single_fea = c_pat_data_np.T
            imp2[i] = np.corrcoef(dis_all_fea, dis_single_fea)[0, 1:]

        return torch.tensor(imp2)

    def FindFeaEmb(
        self,
        ins_emb: torch.tensor,
        pat_val: torch.tensor,
    ):

        if self.importance == None:
            pat_val += torch.randn_like(pat_val)*1e-3
            self.importance = self.Distinguishability(pat_val, num_fea_per_pat=1)
        importance_t = self.importance.t()
        top_k_importance_t = importance_t.max(dim=1)[1]
        best_emb = ins_emb[top_k_importance_t]
        return best_emb, self.importance


def main(args):

    pl.utilities.seed.seed_everything(1)
    info = [str(s) for s in sys.argv[1:]]
    runname = "_".join(["dmt", args.data_name, "".join(info)])

    wandb.init(
        name=runname,
        project="PatEmb" + args.project_name,
        entity="zangzelin",
        mode="offline" if bool(args.offline) else "online",
        save_code=True,
        config=args,
    )
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath='save_checkpoint/', 
        every_n_epochs=args.log_interval,
        filename=args.data_name+'{epoch}'
        )

    model = LitPatNN(
        dataname=args.data_name,
        **args.__dict__,
    )
    early_stop = EarlyStopping(
        monitor="es_monitor", patience=900, verbose=False, mode="max"
    )
    trainer = Trainer(
        gpus=1,
        max_epochs=args.epochs,
        progress_bar_refresh_rate=0,
        # progress_bar_refresh_rate=10,
        callbacks=[
            early_stop, 
            checkpoint_callback
            ],
    )
    trainer.fit(model)

    if args.showmainfig > 0:
        model = model.to('cuda')
        wandb.log({'main/fig': model.up_mainfig(
            model.data, 
            model.ins_emb,
            model.label, model.index, mask=model.mask
            )})

        args.cf_num_data=10000

        if args.data_name == 'InsEmb_Digit':
            pix = 8
        elif args.data_name == 'Mnist':
            pix = 28
        else:
            pix = 0

        if pix > 0:

            cf_index_from = [10,20,30,40,50]
            cf_index_to = [0,1,2]
            num_succ_pic = 0
            num_pix = 0
            for cf_num in cf_index_to:
                img_from, cf = cf_expalain.CF_case_study_abili_Proto(
                            model=model,
                            data_use=model.data,
                            datashow_lat=model.ins_emb,
                            cf_index_from=cf_index_from,
                            cf_index_to=cf_num,
                            num_data=args.cf_num_data,
                        )

                fig_all_img = make_subplots(3, cf.shape[0])
                usefule_cf_mask = cf.sum(axis=1)>1
                # num_succ_pic += usefule_cf_mask.sum()
                # num_pix += (np.abs(cf-img_from)[usefule_cf_mask]>0).sum()

                for j in range(cf.shape[0]):
                    cf_0 = cf[j]
                    cf_from = img_from[j]
                    # cf_to = img_to

                    # cf_to[gpu2np(model.mask<1)] = None
                    cf_from[gpu2np(model.mask<1)] = None
                    cf_0[gpu2np(model.mask<1)] = None

                    fig_all_img.add_trace(go.Heatmap(z=cf_from.reshape(pix, pix)[::-1]), 1, j+1)
                    # fig_all_img.add_trace(go.Heatmap(z=cf_to.reshape(pix, pix)[::-1]), 2, j+1)
                    fig_all_img.add_trace(go.Heatmap(z=np.abs(cf_from-cf_0).reshape(pix, pix)[::-1]), 2, j+1)
                    fig_all_img.add_trace(go.Heatmap(z=cf_0.reshape(pix, pix)[::-1]), 3, j+1)

                log_dict_cf={
                    'cm{}/fig_all_img'.format(cf_num): fig_all_img,
                    # 'cm{}/fig_cf'.format(cf_num): fig_cf,
                }
                wandb.log(log_dict_cf)




if __name__ == "__main__":  
    
    import argparse

    parser = argparse.ArgumentParser(description="*** author")
    parser.add_argument("--offline", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1, metavar="S")
    parser.add_argument("--data_path", type=str, default="/zangzelin/data")
    parser.add_argument("--log_interval", type=int, default=300)
    parser.add_argument("--project_name", type=str, default="test")
    parser.add_argument(
        "--computer", type=str, default=os.popen(
            "git config user.name").read()[:-1]
    )

    # data set param
    parser.add_argument(
        "--data_name",
        type=str,
        default="Mnist",
        choices=[
            'InsEmb_PBMC',
            "OTU",
            "Activity",
            "Gast10k1457",
            # "InsEmb_Car2",
            "PBMCD2638",
            # "InsEmb_Univ",
            # "InsEmb_PBMC",
            "PBMC",
            # "InsEmb_Colon",
            # "InsEmb_Digit",
            # "InsEmb_TPD_579",
            # "InsEmb_TPD_579_ALL_PRO",
            # "InsEmb_TPD_867",
            "Digits",
            "Mnist",
            'Mnist3000',
            'Mnist10000',
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
    parser.add_argument("--detaalpha", type=float, default=1.001)
    parser.add_argument("--l2alpha", type=float, default=20)
    parser.add_argument("--nu", type=float, default=1e-2)
    parser.add_argument("--num_link_aim", type=float, default=0.2)
    # parser.add_argument("--num_fea_aim", type=int, default=128)
    parser.add_argument("--num_fea_aim", type=int, default=60)
    parser.add_argument("--K_plot", type=int, default=40)

    parser.add_argument("--num_fea_per_pat", type=int, default=80)  # 0.5
    # parser.add_argument("--K", type=int, default=3)
    parser.add_argument("--K", type=int, default=5)
    parser.add_argument("--Uniform_t", type=float, default=2)  # 0.3
    parser.add_argument("--Bernoulli_t", type=float, default=-1)  # 0.4
    parser.add_argument("--Normal_t", type=float, default=-1)  # 0.5
    parser.add_argument("--uselabel", type=int, default=0)  # 0.5
    parser.add_argument("--showmainfig", type=int, default=1)  # 0.5

    # train param
    parser.add_argument(
        "--NetworkStructure_1", type=list, default=[-1, 200] + [200] * 1
    )
    parser.add_argument("--NetworkStructure_2",
                        type=list, default=[-1, 500, 80])
    parser.add_argument("--num_pat", type=int, default=8)
    parser.add_argument("--num_latent_dim", type=int, default=2)
    parser.add_argument("--augNearRate", type=float, default=1000)
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1000,
    )
    parser.add_argument("--epochs", type=int, default=50000)
    parser.add_argument("--lr", type=float, default=1e-3, metavar="LR")

    args = pl.Trainer.add_argparse_args(parser)
    args = args.parse_args()

    main(args)
