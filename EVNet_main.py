import functools
import os
import sys

import numpy as np
import plotly.graph_objects as go
import pytorch_lightning as pl
import torch
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torch import nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from plotly.subplots import make_subplots
import local_exp
import swich_exp
import scipy
from torchvision import transforms
import uuid

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


class NN_FCBNRL_MM(nn.Module):
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
        self.alpha = None
        self.stop = False
        self.detaalpha = self.hparams.detaalpha
        self.bestval = 0
        self.aim_cluster = None
        self.importance = None
        self.setup()
        self.wandb_logs = {}
        self.mse = torch.nn.CrossEntropyLoss()
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.hparams.num_pat = min(self.data_train.data.shape[1], self.hparams.num_pat)

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

        if len(self.data_train.data.shape) > 2:
            # crop_size = 32
            # mean=(0.491, 0.482, 0.447)
            # std=(0.247, 0.243, 0.261)
            self.transforms = transforms.AutoAugment(
                transforms.AutoAugmentPolicy.CIFAR10
            )

        # self.m = self.updata_m(init=True)

        self.fea_num = 1
        for i in range(len(self.data_train.data.shape) - 1):
            self.fea_num = self.fea_num * self.data_train.data.shape[i + 1]

        print("fea_num", self.fea_num)
        self.PM_root = nn.Linear(self.fea_num, 1)
        self.PM_root.weight.data = torch.ones_like(self.PM_root.weight.data) / 5
        # nn.Parameter(
        #     torch.tensor(
        #         torch.ones((self.fea_num)) / 5
        #     )
        # )

    def forward_fea(self, x):

        # lat = torch.zeros(x.shape).to(x.device)
        self.mask = self.PM_root.weight.reshape(-1) > 0.1
        # for i in range(self.hparams.num_pat):
        if self.alpha is not None:
            lat = x * ((self.PM_root.weight.reshape(-1)) * self.mask)
        else:
            lat = x * ((self.PM_root.weight.reshape(-1)) * self.mask).detach()
        lat1 = self.model_pat(lat)
        lat3 = lat1
        for i, m in enumerate(self.model_b):
            lat3 = m(lat3)
        return lat1, lat1, lat3

    def forward_exp(self, x):

        x_ = x * ((self.PM_root.weight.reshape(-1)) * self.mask)
        lat1 = self.model_pat(x_.float())
        lat3 = lat1
        for i, m in enumerate(self.model_b):
            lat3 = m(lat3)
        return lat1, lat1, lat3

    def forward(self, x):
        return self.forward_fea(x)

    def forward_save(self, x):
        return self.forward_exp(x)

    def predict(self, x):
        x = torch.tensor(x.to_numpy())
        return gpu2np(self.forward_simi(x))

    def predict_(self, x):
        x_b = torch.zeros((x.shape[0], self.mask.shape[0]))
        x_b[:, self.mask > 0] = torch.tensor(x).float()
        sim = gpu2np(self.forward_simi(x_b)).reshape(-1, 1)
        out = np.concatenate([sim, 1 - sim], axis=1)
        return out

    def predict_lime_one_epoch(self, x):
        x_b = torch.zeros((x.shape[0], self.mask.shape[0])).to(self.mask.device)
        x_b[:, self.mask > 0] = torch.tensor(x).float().to(self.mask.device)
        lat = gpu2np(self(x_b)[2])
        predict_r = []
        for i in range(self.cluster_centers.shape[0]):
            predict_r.append(
                np.exp(
                    -1
                    * np.linalg.norm(
                        lat - self.cluster_centers[i], axis=1, keepdims=True
                    )
                )
            )
        predict = np.concatenate(predict_r, axis=1)

        return scipy.special.softmax(predict, axis=1)

    def predict_lime_g(self, x):

        lat = self.forward_exp(x)[2]
        predict_r = []
        for i in range(self.cluster_centers.shape[0]):
            dis = torch.norm(lat - self.cluster_centers[i], dim=1, keepdim=True)
            # if self.aim_cluster and i == self.aim_cluster:
            #     dis = dis * self.cluster_rescale[i]
            predict_r.append(torch.exp(-2 * dis))
        predict = torch.cat(predict_r, dim=1)

        return torch.softmax(predict, dim=1)

    def predict_lime_g_for_numpy(self, x):
        x_base = torch.zeros(size=(x.shape[0], self.mask.shape[0]))
        # print(type(x) == np.float32)
        if type(x) == np.ndarray:
            x_base[:, self.mask] = torch.tensor(x).float()
        else:
            x_base[:, self.mask] = x.float()
        # x_base[:, self.mask] = x.float()
        predict = self.predict_lime_g(x_base.to(self.device))
        predict[:, self.aim_cluster] = predict[:, self.aim_cluster] * 0.95
        # print(gpu2np(predict))
        return gpu2np(predict)

    def forward_simi(self, x):
        x = torch.tensor(x).to(self.mask.device)
        out = self.forward_fea(x)[2]
        dis = torch.norm(out - torch.tensor(self.cf_aim).to(x.device), dim=1)
        return torch.exp(-1 * dis).reshape(-1)

    def training_step(self, batch, batch_idx):
        index = batch.to(self.device)
        # augmentation
        data1 = self.data_train.data[index]
        data2 = self.augmentation_warper(index, data1)
        data = torch.cat([data1, data2])
        data = data.reshape(data.shape[0], -1)

        # label1 = self.data_train.label[index]
        # label2 = self.data_train.label[index]
        # label2 = self.augmentation(index, label1)
        # label = torch.cat([label1, label2])

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

        # pre_dict = self.forward_classi(lat.detach())
        # loss_mse = self.mse(pre_dict, label,)

        self.wandb_logs = {
            # "loss_mse": loss_mse,
            "loss_topo": loss_topo,
            "lr": self.trainer.optimizers[0].param_groups[0]["lr"],
            "epoch": self.current_epoch,
            # "T": self.t_list[self.current_epoch],
        }

        loss_l2 = 0
        if self.current_epoch >= 300 and batch_idx == 0:
            if self.alpha is None:
                # print("--->")
                self.alpha = loss_topo.detach().item() / (
                    self.Cal_Sparse_loss(
                        self.PM_root.weight.reshape(-1),
                    ).detach()
                    * self.hparams.l2alpha
                )

            N_Feature = np.sum(gpu2np(self.mask) > 0)
            if N_Feature > self.hparams.num_fea_aim:
                loss_l2 = self.Cal_Sparse_loss(self.PM_root.weight.reshape(-1))
                self.alpha = self.alpha * self.detaalpha
                loss_topo += (loss_l2) * self.alpha
        return loss_topo

    def validation_step(self, batch, batch_idx):
        # augmentation
        if (self.current_epoch + 1) % self.hparams.log_interval == 0:
            index = batch.to(self.device)
            data = self.data_train.data[index]
            data = data.reshape(data.shape[0], -1)
            pat, mid, lat = self(data)

            return (
                gpu2np(data),
                gpu2np(pat),
                gpu2np(lat),
                np.array(self.data_train.label.cpu())[gpu2np(index)],
                gpu2np(index),
            )

    def Cal_Sparse_loss(self, PatM):
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

            if self.alpha is not None and N_Feature <= self.hparams.num_fea_aim:
                ecb_e_train = ecb.Eval(input=data, latent=ins_emb, label=label, k=10)
                data_test = self.data_test.data
                label_test = self.data_test.label
                _, _, lat_test = self(data_test)
                ecb_e_test = ecb.Eval(
                    input=gpu2np(data_test),
                    latent=gpu2np(lat_test),
                    label=gpu2np(label_test),
                    k=10,
                )

                self.wandb_logs.update(
                    {
                        "epoch": self.current_epoch,
                        "alpha": self.alpha,
                        "metric/#link": N_link,
                        "metric/#Feature": N_Feature,
                        "SVC_train": ecb_e_train.E_Classifacation_SVC(),  # SVC= ecb_e.E_Classifacation_SVC(),
                        "SVC_test": ecb_e_test.E_Classifacation_SVC(),  # SVC= ecb_e.E_Classifacation_SVC(),
                    }
                )

                ec.ShowEmb(ins_emb, self.data_train.labelstr, index)
            else:
                self.wandb_logs.update(
                    {
                        # "ShowEmb": go.Figure(data=ec.ShowEmb_return_fig(
                        #     ins_emb, self.data_train.labelstr, index)),
                        "epoch": self.current_epoch,
                        "alpha": self.alpha,
                        "metric/#link": N_link,
                        "metric/#Feature": N_Feature,
                    }
                )
                wandb.log(
                    {
                        "main_easy/fig_easy": self.up_mainfig_emb(
                            data, ins_emb, label, index, mask=gpu2np(self.mask)
                        )
                    }
                )
            if self.hparams.save_checkpoint:
                np.save(
                    "save_checkpoint/"
                    + self.hparams.data_name
                    + "={}".format(self.current_epoch),
                    gpu2np(self.PM_root.weight.data),
                )

            if N_Feature <= self.hparams.num_fea_aim:
                self.stop = True
            else:
                self.stop = False

            if self.wandb_logs is not None:
                wandb.log(self.wandb_logs)

        else:
            self.log("SVC", 0)

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
        # import pdb; pdb.set_trace()
        dataset_f = getattr(data_base, self.dataname + "Dataset")
        self.data_train = dataset_f(
            data_name=self.hparams.data_name,
            train=True,
            datapath=self.hparams.data_path,
        )
        if len(self.data_train.data.shape) == 2:
            self.data_train.cal_near_index(
                device='cuda',
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
            batch_size=min(self.hparams.batch_size, self.data_train.data.shape[0]),
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.data_train,
            batch_size=min(self.hparams.batch_size, self.data_train.data.shape[0]),
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
        struc_model_b = NetworkStructure_2 + [2]
        struc_model_b[0] = num_fea_per_pat

        m_l = []
        for i in range(len(struc_model_pat) - 1):
            m_l.append(
                NN_FCBNRL_MM(
                    struc_model_pat[i],
                    struc_model_pat[i + 1],
                )
            )
        model_pat = nn.Sequential(*m_l)

        model_b = nn.ModuleList()
        for i in range(len(struc_model_b) - 1):
            if i != len(struc_model_b) - 2:
                model_b.append(NN_FCBNRL_MM(struc_model_b[i], struc_model_b[i + 1]))
            else:
                model_b.append(
                    NN_FCBNRL_MM(struc_model_b[i], struc_model_b[i + 1], use_RL=False)
                )

        print(model_pat)
        print(model_b)
        return model_pat, model_b

    def augmentation_warper(self, index, data1):
        if len(data1.shape) == 2:
            return self.augmentation(index, data1)
        else:
            return self.augmentation_img(index, data1)

    def augmentation_img(self, index, data):
        # aug = []
        # for i in range(data.shape[0]):
        #     aug.append(
        #         self.transforms(data.permute(0,3,1,2)).reshape(1,-1)
        #         )
        return self.transforms(data.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

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

    def up_mainfig(
        self,
        data,
        ins_emb,
        label,
        index,
        mask,
        n_clusters=10,
        num_cf_example=2,
        explevel=3,
    ):

        if self.data_train.data.shape[0] > 20000:
            self.rand_index = torch.randperm(self.data_train.data.shape[0])[:10000]
        else:
            self.rand_index = torch.tensor(
                [i for i in range(self.data_train.data.shape[0])]
            )
        if self.hparams.data_name == "Digits":
            pix = 8
        elif self.hparams.data_name == "Mnist":
            pix = 28
        else:
            pix = 0

        fig = make_subplots(
            rows=2,
            cols=3,
            column_widths=[0.3, 0.4, 0.3],
            row_heights=[0.6, 0.4],
            vertical_spacing=0.05,
            horizontal_spacing=0.05,
            specs=[
                [{"type": "xy"}, {"type": "xy"}, {"type": "xy"}],
                [{"type": "xy"}, {"type": "sankey"}, {"type": "xy"}],
            ],
            subplot_titles=(
                "sample Embedding",
                "sample-feature-pattern Embedding",
                "Switching Embedding",
                "Global Explainability",
                "Local Explainability",
                "Switching Explainability",
            ),
        )

        # clustering
        label_pesodu, cluster_centers = local_exp.Keams_clustering(ins_emb, n_clusters)
        self.cluster_centers = torch.tensor(cluster_centers).to(self.mask.device)

        from sklearn.preprocessing import MinMaxScaler

        global_importance_raw = np.copy(gpu2np(self.PM_root.weight.reshape(-1)))
        global_importance_raw[~gpu2np(self.mask)] = 0.1
        global_importance = MinMaxScaler().fit_transform(
            global_importance_raw[:, None]
        )[:, 0]

        try:
            fea_list_all = self.data_train.feature_name
        except BaseException:
            print("use the index as the feature name")
            fea_list_all = np.array(
                ["f{}".format(i) for i in range(global_importance.shape[0])]
            )
        global_importance = global_importance[gpu2np(self.mask)]
        fea_list = fea_list_all[gpu2np(self.mask)]

        shap_values, fea_emb, _ = local_exp.LocalExplainability(
            data=data,
            model=self,
            num_s_shap=100,
        )
        if explevel > 2:
            img_from_list, cf_list = swich_exp.SwichExplainability(
                model=self,
                data_use=data,
                label_pesodu=label_pesodu,
                cluster_number=n_clusters,
                cf_max_iterations=500,
                pix=pix,
            )

        try:
            label_name_list = self.data_train.label_name_list
        except BaseException:
            label_name_list = None

        # add subfigs
        sf1_1 = ec.Plot_subfig_1_1(ins_emb, label, index, label_name_list=label_name_list)
        sf2_1 = ec.Plot_subfig_2_1(
            global_importance=global_importance, fea_list=fea_list
        )
        sf1_2 = ec.Plot_subfig_1_2(
            ins_emb=ins_emb,
            label_pesodu=label_pesodu,
            cluster_centers=cluster_centers,
            # shap_values=shap_values,
        )
        sf2_2 = ec.Plot_subfig_2_2(
            shap_values,
            fea_list_all=fea_list_all.tolist(),
        )
        sf1_3 = []
        if explevel > 2:
            sf1_3_c01, change_dict_for_ij_list, _ = ec.Plot_subfig_1_3(
                model=self,
                ins_emb=ins_emb,
                cf_list=cf_list,
                img_from_list=img_from_list,
                pix=pix,
            )

            sf1_3 += sf1_3_c01
            sf2_3 = ec.Plot_subfig_2_3(
                model=self,
                change_dict_for_ij_list=change_dict_for_ij_list,
                img_from_list=img_from_list,
                cf_list=cf_list,
                n_clusters=n_clusters,
                fea_list_all=fea_list_all,
            )
            sf1_3 += sf1_2[: n_clusters + 1]

        # add traces
        fig.add_traces(sf1_1, rows=[1] * len(sf1_1), cols=[1] * len(sf1_1))
        fig.add_traces(sf2_1, rows=[2] * len(sf2_1), cols=[1] * len(sf2_1))
        fig.add_traces(sf1_2, rows=[1] * len(sf1_2), cols=[2] * len(sf1_2))
        sankey_index_sart = len(fig.data)
        fig.add_traces(sf2_2, rows=[2] * len(sf2_2), cols=[2] * len(sf2_2))
        sankey_index_end = len(fig.data)
        if explevel > 2:
            flow_sart = len(fig.data) + 1
            fig.add_traces(
                sf1_3,
                rows=[1] * len(sf1_3),
                cols=[3] * len(sf1_3),
            )
            flow_end = flow_sart + len(sf1_3_c01) - 1
            fig.add_traces(
                sf2_3,
                rows=[2] * len(sf2_3),
                cols=[3] * len(sf2_3),
            )

            fig = ec.add_button(
                fig, sankey_index_sart, sankey_index_end, flow_sart, flow_end, n_clusters
            )

        ec.Plot_case_study(
            data=data,
            mask=mask,
            pix=pix,
            num_s_shap=data.shape[0] // 10,
            shap_values_abs=shap_values,
        )

        fig.update_layout(
            height=1200,
            width=2000,
            showlegend=False,
            title_text="ENV Result of {}".format(self.hparams.data_name),
        )

        fig.write_html(
            "save_html/{}_{}_{}.html".format(
                self.hparams.data_name, self.current_epoch, str(uuid.uuid1())
            )
        )

        return fig

    def up_mainfig_case(
        self,
        data,
        ins_emb,
        label,
        index,
        mask,
        top_cluster=10,
        n_clusters=10,
        num_cf_example=2,
        explevel=3,
    ):

        if self.data_train.data.shape[0] > 20000:
            self.rand_index = torch.randperm(
                self.data_train.data.shape[0])[:10000]
        else:
            self.rand_index = torch.tensor(
                [i for i in range(self.data_train.data.shape[0])]
            )
        if self.hparams.data_name == "Digits":
            pix = 8
        elif self.hparams.data_name == "Mnist":
            pix = 28
        else:
            pix = 0

        fig = make_subplots(
            rows=2,
            cols=3,
            column_widths=[0.3, 0.4, 0.3],
            row_heights=[0.6, 0.4],
            vertical_spacing=0.05,
            horizontal_spacing=0.05,
            specs=[
                [{"type": "xy"}, {"type": "xy"}, {"type": "xy"}],
                [{"type": "xy"}, {"type": "sankey"}, {"type": "xy"}],
            ],
            subplot_titles=(
                "sample Embedding",
                "sample-feature-pattern Embedding",
                "Switching Embedding",
                "Global Explainability",
                "Local Explainability",
                "Switching Explainability",
            ),
        )

        # clustering
        label_pesodu, cluster_centers = local_exp.Keams_clustering(ins_emb, n_clusters)
        self.cluster_centers = torch.tensor(cluster_centers).to(self.mask.device)

        from sklearn.preprocessing import MinMaxScaler

        global_importance_raw = np.copy(gpu2np(self.PM_root.weight.reshape(-1)))
        global_importance_raw[~gpu2np(self.mask)] = 0.1
        global_importance = MinMaxScaler().fit_transform(
            global_importance_raw[:, None]
        )[:, 0]

        try:
            fea_list_all = self.data_train.feature_name
        except BaseException:
            print("use the index as the feature name")
            fea_list_all = np.array(
                ["f{}".format(i) for i in range(global_importance.shape[0])]
            )
        global_importance = global_importance[gpu2np(self.mask)]
        fea_list = fea_list_all[gpu2np(self.mask)]

        shap_values_abs, fea_emb, shap_values = local_exp.LocalExplainability(
            data=data,
            model=self,
            num_s_shap=200,)
        if explevel > 2:
            img_from_list, cf_list = swich_exp.SwichExplainability(
                model=self,
                data_use=data,
                label_pesodu=label_pesodu,
                cluster_number=n_clusters,
                cf_max_iterations=5,
                cf_example_number=10,
                pix=pix,
                top_cluster=top_cluster,
            )

        try:
            label_name_list = self.data_train.label_name_list
        except BaseException:
            label_name_list = None

        # add subfigs
        sf1_1 = ec.Plot_subfig_1_1(
            ins_emb, label, index,
            label_name_list=label_name_list)
        sf2_1 = ec.Plot_subfig_2_1(
            global_importance=global_importance, fea_list=fea_list
        )
        sf1_2 = ec.Plot_subfig_1_2(
            ins_emb=ins_emb,
            label_pesodu=label_pesodu,
            cluster_centers=cluster_centers,
            # shap_values=shap_values,
        )
        sf2_2 = ec.Plot_subfig_2_2(
            shap_values_abs,
            fea_list_all=fea_list_all.tolist(),
        )
        sf1_3 = []
        if explevel > 2:
            sf1_3_c01, change_dict_for_ij_list, _ = ec.Plot_subfig_1_3(
                model=self,
                ins_emb=ins_emb,
                cf_list=cf_list,
                img_from_list=img_from_list,
                pix=pix,
            )

            sf1_3 += sf1_3_c01
            sf2_3 = ec.Plot_subfig_2_3(
                model=self,
                change_dict_for_ij_list=change_dict_for_ij_list,
                img_from_list=img_from_list,
                cf_list=cf_list,
                n_clusters=n_clusters,
                fea_list_all=fea_list_all,
            )
            sf1_3 += sf1_2[: n_clusters + 1]

        # add traces
        fig.add_traces(sf1_1, rows=[1] * len(sf1_1), cols=[1] * len(sf1_1))
        fig.add_traces(sf2_1, rows=[2] * len(sf2_1), cols=[1] * len(sf2_1))
        fig.add_traces(sf1_2, rows=[1] * len(sf1_2), cols=[2] * len(sf1_2))
        sankey_index_sart = len(fig.data)
        fig.add_traces(sf2_2, rows=[2] * len(sf2_2), cols=[2] * len(sf2_2))
        sankey_index_end = len(fig.data)
        if explevel > 2:
            flow_sart = len(fig.data) + 1
            fig.add_traces(
                sf1_3,
                rows=[1] * len(sf1_3),
                cols=[3] * len(sf1_3),
            )
            flow_end = flow_sart + len(sf1_3_c01) - 1
            fig.add_traces(
                sf2_3,
                rows=[2] * len(sf2_3),
                cols=[3] * len(sf2_3),
            )

            fig = ec.add_button(
                fig, sankey_index_sart, sankey_index_end,
                flow_sart, flow_end, n_clusters
            )

        # fig.add_traces(sf2_2, rows=[2] * len(sf2_2), cols=[2] * len(sf2_2))
        fig_22 = go.Figure(data=sf2_2)
        wandb.log({'fig22': fig_22})

        # ec.Plot_case_study(
        #     data=data,
        #     mask=mask,
        #     pix=pix,
        #     num_s_shap=data.shape[0] // 10,
        #     shap_values=shap_values,
        # )

        fig.update_layout(
            height=1200,
            width=2000,
            showlegend=False,
            title_text="ENV Result of {}".format(self.hparams.data_name),
        )

        fig.write_html(
            "save_html/{}_{}_{}.html".format(
                self.hparams.data_name, self.current_epoch, str(uuid.uuid1())
            )
        )

        shap_values_all = []
        for i in range(len(shap_values)):
            shap_values_all.append(
                shap_values[i].reshape((1, *shap_values[0].shape))
                )
        shap_values_all = np.concatenate(shap_values_all)

        ec.Plot_case_study(
            data, mask, pix,
            num_s_shap=10,
            shap_values=shap_values_all,
            shap_values_abs=shap_values_abs,
            label_pesodu=label_pesodu,
            )
        # px.imshow()

        return fig


    def up_mainfig_emb(
        self, data, ins_emb,
        label, index, mask,
        n_clusters=10, num_cf_example=2,
    ):
        color = np.array(label)
        import plotly.express as px

        fig = px.scatter(
            x=ins_emb[:, 0], y=ins_emb[:, 1], color=[str(c) for c in color]
        )
        return fig




def main(args):

    pl.utilities.seed.seed_everything(1)
    info = [str(s) for s in sys.argv[1:]]
    runname = "_".join(["dmt", args.data_name, "".join(info)])
    callbacks_list = []

    wandb.init(
        name=runname,
        project="EVNet" + args.project_name,
        entity="zangzelin",
        mode="offline" if bool(args.offline) else "online",
        save_code=True,
        config=args,
    )

    model = LitPatNN(dataname=args.data_name,**args.__dict__,)

    trainer = Trainer(
        gpus=1,
        max_epochs=args.epochs,
    )
    print("start fit")
    trainer.fit(model)
    print("end fit")


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description="*** author")

    # data set param
    parser.add_argument(
        "--data_name",
        type=str,
        default="Mnist",
        choices=[
            "InsEmb_PBMC",
            "OTU",
            "Activity",
            "Gast10k1457",
            "PBMCD2638",
            "PBMC",
            "InsEmb_TPD_579_ALL_PRO",
            "InsEmb_TPD_579_ALL_PRO5C",
            "YONGJIE_UC",
            "Digits",
            "Mnist",
            "Mnist3000",
            "Mnist10000",
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
    parser.add_argument("--offline", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1, metavar="S")
    parser.add_argument("--data_path", type=str, default="/zangzelin/data")
    parser.add_argument("--log_interval", type=int, default=300)
    parser.add_argument("--project_name", type=str, default="test")
    parser.add_argument(
        "--computer", type=str,
        default=os.popen("git config user.name").read()[:-1]
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
    parser.add_argument("--l2alpha", type=float, default=10)
    parser.add_argument("--nu", type=float, default=1e-2)
    parser.add_argument("--num_link_aim", type=float, default=0.2)
    # parser.add_argument("--num_fea_aim", type=int, default=128)
    parser.add_argument("--num_fea_aim", type=int, default=50)
    parser.add_argument("--K_plot", type=int, default=40)
    parser.add_argument("--save_checkpoint", type=int, default=0)

    parser.add_argument("--num_fea_per_pat", type=int, default=80)  # 0.5
    # parser.add_argument("--K", type=int, default=3)
    parser.add_argument("--K", type=int, default=5)
    parser.add_argument("--Uniform_t", type=float, default=1)  # 0.3
    parser.add_argument("--Bernoulli_t", type=float, default=-1)
    parser.add_argument("--Normal_t", type=float, default=-1)
    parser.add_argument("--uselabel", type=int, default=0)
    parser.add_argument("--showmainfig", type=int, default=1)

    # train param
    parser.add_argument(
        "--NetworkStructure_1", type=list, default=[-1, 200] + [200] * 5
    )
    parser.add_argument("--NetworkStructure_2", type=list, default=[-1, 500, 80])
    parser.add_argument("--num_pat", type=int, default=8)
    parser.add_argument("--num_latent_dim", type=int, default=2)
    parser.add_argument("--augNearRate", type=float, default=1000)
    parser.add_argument("--explevel", type=int, default=3)
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1000,
    )
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=1e-3, metavar="LR")

    args = pl.Trainer.add_argparse_args(parser)
    args = args.parse_args()

    main(args)
