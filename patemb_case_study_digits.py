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
import plotly.express as px

import uuid

import cf_expalain
import eval.eval_core as ec
import eval.eval_core_base as ecb
import Loss.dmt_loss_aug as dmt_loss_aug
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
    return res


class MyLinearBMM(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        channel: int,
        bias: bool = True,
        rescon: bool = True,
    ) -> None:
        super(nn.Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rescon = rescon
        self.channel = channel
        self.weight = Parameter(
            torch.Tensor(
                channel,
                in_features,
                out_features,
            )
        )
        if bias:
            self.bias = Parameter(torch.Tensor(channel, out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if len(input.shape) != len(self.weight.shape):
            input = input.reshape((*input.shape, 1)).expand(
                (
                    *input.shape,
                    self.channel,
                )
            )
        out = torch.bmm(
            input.permute(2, 0, 1), self.weight.to(input.device)
        ) + self.bias[:, None, :].to(input.device)
        out = out.permute(1, 2, 0)
        if self.rescon:
            out += input
        return out

    def extra_repr(self) -> str:
        return "in_features={}, out_features={}, bias={}, rescon={}, channel{}".format(
            self.in_features,
            self.out_features,
            self.bias is not None,
            self.rescon,
            self.channel,
        )


class NN_FCBNRL_BMM(nn.Module):
    # def forward(self, input: torch.Tensor) -> torch.Tensor:
    #     return torch.nn.functional.linear(input, self.weight, self.bias)
    def __init__(self, in_dim, out_dim, channel=8, use_RL=True, rescon=False):
        super(NN_FCBNRL_BMM, self).__init__()
        m_l = []
        m_l.append(MyLinearBMM(
            in_dim, out_dim, channel=channel, rescon=rescon))
        if use_RL:
            m_l.append(nn.BatchNorm1d(out_dim))
            m_l.append(nn.LeakyReLU(0.1))
        self.block = nn.Sequential(*m_l)

    def forward(self, x):
        return self.block(x)


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
        self.new_m = None
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

        self.m = self.updata_m(init=True)
        self.PM_root = nn.Parameter(
            torch.tensor(
                torch.ones((self.data_train.data.shape[1],
                            self.hparams.num_pat)) / 5
            )  # +torch.tensor(m).float()
        )



    def updata_m(self, init=False):

        if init:
            return torch.ones((
                self.data_train.data.shape[1],
                self.hparams.num_pat)).to("cuda")-1
        else:
            fea_fea_emb, fea_emb_importance = self.FindFeaEmb(
                ins_emb=self(self.data_train.data)[2].detach().cpu(),
                pat_val=self.data_train.data.detach().cpu(),
            )

            fea_fea_emb = fea_fea_emb.detach().cpu().numpy()
            m = np.zeros(shape=(
                self.data_train.data.shape[1], self.hparams.num_pat)) - 0.1
            kmeans = SpectralClustering(
                n_clusters=self.hparams.num_pat,
                assign_labels="discretize",
                random_state=0,
                eigen_solver="arpack",
                affinity="nearest_neighbors",
            ).fit(fea_fea_emb)

            for i, k in enumerate(kmeans.labels_):
                m[i, k] = 0.0

            return torch.tensor(m).float().to("cuda")

    def updata_mask(
        self,
    ):
        PatternMatrix = self.PM_root + self.m
        maskt = PatternMatrix > self.t
        mask0 = PatternMatrix >= PatternMatrix.max(dim=0)[0].reshape(1, -1)
        self.mask = ((maskt + mask0) > 0).to(self.device)

    def forward_fea(self, x):

        lat = torch.zeros((*x.shape, self.hparams.num_pat)).to(x.device)
        for i in range(self.hparams.num_pat):
            if self.alpha is not None:
                lat[:, :, i] = x * (
                    (self.PM_root + self.m)[:, i] * self.mask[:, i])
            else:
                lat[:, :, i] = x * (
                    (self.PM_root + self.m)[:, i] * self.mask[:, i]
                    ).detach()
        
        y = self.model_pat(lat)
        lat1 = y.reshape(y.shape[0], -1)

        lat3 = lat1
        for i, m in enumerate(self.model_b):
            lat3 = m(lat3)
        return lat1, lat1, lat3

    def forward(self, x):
        return self.forward_fea(x)

    def predict(self, x):
        x = torch.tensor(x.to_numpy())
        return gpu2np(self.forward_simi(x))

    def predict_(self, x):
        x_b = torch.zeros((x.shape[0], self.mask.shape[0]))
        x_b[:,self.mask.sum(dim=1)>0] = torch.tensor(x).float()
        sim = gpu2np(self.forward_simi(x_b)).reshape(-1,1)
        out = np.concatenate([sim, 1-sim], axis=1)
        return out

    def forward_simi(self, x):
        x = x.to(self.m.device)
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

        # if batch_idx == 0:
        self.updata_mask()

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
        if torch.rand(1) > 0.8 and self.current_epoch > 600:
            if self.alpha is None:
                print("if self.alpha is None:")
                self.alpha = loss_topo.detach().item() / (
                    self.Cal_Sparse_loss(
                        self.PM_root + self.m,
                        self.mask,
                        self.hparams.num_pat,
                    ).detach()
                    * self.hparams.l2alpha
                )
                self.m = self.updata_m()
                self.updata_mask()

            N_Feature = np.sum(gpu2np(self.mask).sum(axis=1) > 0)
            if N_Feature > self.hparams.num_fea_aim:
                loss_l2 = self.Cal_Sparse_loss(
                    self.PM_root + self.m,
                    self.mask,
                    self.hparams.num_pat,
                )
                self.alpha = self.alpha * self.detaalpha
                loss_topo += (loss_l2) * self.alpha
        return loss_topo

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
        loss_l2 = 0
        for p in range(num_pat):
            c_pat_for_PM = PatM[:, p]
            mean_mask = mask.sum() / mask.shape[1]
            loss_l2 += torch.abs(c_pat_for_PM).mean() * (
                (mask[:, p].sum() / mean_mask) ** 2
            )
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


            N_link = np.sum(gpu2np(self.mask))
            feature_use_bool = gpu2np(self.mask).sum(axis=1) > 0
            N_Feature = np.sum(feature_use_bool)
            
            if self.alpha is not None:
                ecb_e = ecb.Eval(
                    input=data,
                    latent=ins_emb,
                    label=label,
                    k=10
                    )
                # SVC = ecb_e.E_Classifacation_SVC()

                self.wandb_logs.update(
                    {
                        "epoch": self.current_epoch,
                        "alpha": self.alpha,
                        "metric/#link": N_link,
                        "metric/#Feature": N_Feature,
                        "metric/mask": go.Figure(data=go.Heatmap(
                            z=gpu2np(self.mask).astype(np.float32))),
                        "metric/PM_root": go.Figure(
                            data=go.Heatmap(z=gpu2np(self.PM_root))
                        ),
                        "metric/m": go.Figure(data=go.Heatmap(z=gpu2np(self.m))),
                        "metric/PatMatrix+mask": go.Figure(
                            data=go.Heatmap(
                                z=gpu2np(self.PM_root + self.m) * gpu2np(self.mask)
                            )
                        ),
                        "metric/PatMatrix": go.Figure(
                            data=go.Heatmap(z=gpu2np(self.PM_root + self.m))
                        ),
                        'SVC': ecb_e.E_Classifacation_SVC(), #SVC= ecb_e.E_Classifacation_SVC(),
                        'main/fig': self.up_mainfig(
                            data, mid_old, mid_old, ins_emb, label, index,
                            mask=self.mask
                            )
                    }
                )
                # self.log('SVC', SVC_value)
                # if self.current_epoch > self.hparams.epochs-300:
                # self.wandb_logs.update(
                #     self.up_fig(data, mid_old, mid_old, ins_emb, label, index)
                # )
            
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
            

    def up_mainfig(self, data, mid, mid_old, ins_emb, label, index, mask):

        # fig = make_subplots(rows=1, cols=2)
        # import plotly.graph_objects as go
        fea_emb, fea_emb_importance = self.FindFeaEmb(
            ins_emb=torch.tensor(ins_emb),
            pat_val=torch.tensor(data),
        )
        pat_emb = []
        for i in range(self.mask.shape[1]):
            mask_c = self.mask[:, i].detach().cpu().numpy()
            pat_emb.append(fea_emb[mask_c].mean(axis=0).reshape(1, -1))
        pat_emb = torch.tensor(np.concatenate(pat_emb))

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

        fig.add_trace(
            ec.ShowEmb_return_fig(ins_emb, self.data_train.labelstr, index),
            row=1, col=1,
        )

        fig = ec.ShowEmbInsFeaPat_returen_fig(
            ins_emb, pat_emb, pat_emb, fea_emb,
            fea_emb, label, mask=self.mask, fig=fig,
            row=1, col=2,
        )

        # try: row=1, col=3,
        # try:
        fig = cf_expalain.CF(
                self,
                # cf_index_from=self.data_train.data.shape[0]//2,
                cf_index_from=500,
                cf_index_to=200,
                n_cf=4,
                fig_main=fig
            )
        # except:
        #     print('---------')

        fig.add_trace(
            ec.Show_global_importance_Zelin_return_fig(
                gpu2np(self.PM_root + self.m) * gpu2np(self.mask)),
            row=2, col=1,
        )

        fig.add_trace(
            ec.ShowSankey_Zelin_return_fig(mask),
            row=2, col=2,
        )

        fig.update_layout(
            height=1200, width=2000,
            showlegend=False,
            title_text="ENV Result of {}".format(self.hparams.data_name)
            )
        
        fig.write_html('save_html/{}_{}_{}.html'.format(
            self.hparams.data_name, self.current_epoch, str(uuid.uuid1())))

        return fig

    def up_fig(self, data, mid, mid_old, ins_emb, label, index):

        wandb_logs = {}
        fea_emb, fea_emb_importance = self.FindFeaEmb(
            ins_emb=torch.tensor(ins_emb),
            pat_val=torch.tensor(data),
        )
        pat_emb = []
        for i in range(self.mask.shape[1]):
            mask_c = self.mask[:, i].detach().cpu().numpy()
            pat_emb.append(fea_emb[mask_c].mean(axis=0).reshape(1, -1))
        pat_emb = torch.tensor(np.concatenate(pat_emb))

        wandb_logs.update(
            ec.ShowEmbInsFeaPat(
                ins_emb, pat_emb, pat_emb, fea_emb,
                fea_emb, label, mask=self.mask))

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
        struc_model_b[0] = self.hparams.num_pat * num_fea_per_pat

        m_l = []
        for i in range(len(struc_model_pat) - 1):
            m_l.append(
                NN_FCBNRL_BMM(
                    struc_model_pat[i],
                    struc_model_pat[i + 1],
                    channel=self.hparams.num_pat,
                    rescon=(i != 0) and (i != len(struc_model_pat) - 2),
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
        import pdb; pdb.set_trace()
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

        imp = torch.zeros(size=(pat_data.shape[0], num_pat))
        for j in range(num_pat):
            pat_index_s = j * num_fea_per_pat
            pat_index_e = j * num_fea_per_pat + num_fea_per_pat
            pat_data_for_pat_c = pat_data[:, pat_index_s:pat_index_e]

            data_mid = pat_data[:, pat_index_s:pat_index_e]
            data_local = pat_data_for_pat_c[local_index][:, :, 0]
            local_score = torch.abs(
                torch.sort(torch.abs(data_mid - data_local))[1]
                - torch.arange(0, K_dis).float()
            ).mean(dim=1)
            imp[:, j] = local_score

        return 1 - imp

    def FindFeaEmb(
        self,
        ins_emb: torch.tensor,
        pat_val: torch.tensor,
    ):

        importance = self.Distinguishability(pat_val, num_fea_per_pat=1)
        importance_t = importance.t()
        top_k_importance_t = importance_t.max(dim=1)[1]
        best_emb = ins_emb[top_k_importance_t]
        return best_emb, importance


def main(args):

    num_data = 20000
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

    model = LitPatNN.load_from_checkpoint(
        'save_checkpoint_use/InsEmb_Digitepoch=1799.ckpt').to('cuda')
    model.m = torch.tensor(np.load(
        'save_checkpoint_use/InsEmb_Digit_1799.npy')).to('cuda')
    # model = LitPatNN.load_from_checkpoint(
    #     'save_checkpoint_use/Mnistepoch=899.ckpt').to('cuda')
    # model.m = torch.tensor(np.load(
    #     'save_checkpoint_use/Mnist_899.npy')).to('cuda')
    model.eval()

    model.updata_mask()
    pat, mid, lat = model(model.data_train.data[:num_data])
    lat = gpu2np(lat)
    
    print(model.data_train.label[:20])
    # input()

    log_dict = {
        'embeding': px.scatter(
            x=lat[:, 0],
            y=lat[:, 1],
            color=[str(i) for i in gpu2np(model.data_train.label[:num_data]).tolist()],
        ),
        # 'c/mask_sankey':go.Figure(data=[
        #     ec.ShowSankey_Zelin_return_fig(model.mask)
        #     ]),
    }

    pattern_colored_pix = np.zeros(8*8)-1
    for fea in range(model.mask.shape[0]):
        for pat in range(model.mask.shape[1]):
            if model.mask[fea, pat]:
                pattern_colored_pix[fea] = pat
    
    # log_dict['c/pattern_colored_pix'] = go.Figure(
    #     data=go.Heatmap(z=pattern_colored_pix.reshape(28,28)))
    log_dict['c/pattern_colored_pix_imshow'] = px.imshow(
        pattern_colored_pix.reshape(8, 8))
    
    wandb.log(log_dict)

    cf_index_from = [1, 2, 4, 5, 6, 7, 8, 12, 16, 19]
    cf_index_to = [23, 24, 25, 26, 27]
    num_succ_pic = 0
    num_pix = 0
    for cf_num in cf_index_to:
        img_to, img_from, cf = cf_expalain.CF_case_study_abili_Proto(
                    model=model,
                    cf_index_from=cf_index_from,
                    cf_index_to=cf_num,
                    num_data=20000,
                    beta=args.beta,
                    kappa=args.kappa,
                    theta=args.theta,
                    cf_max_iterations=1000,
                )

        fig_all_img = make_subplots(4, cf.shape[0])
        usefule_cf_mask = cf.sum(axis=1)>1
        num_succ_pic += usefule_cf_mask.sum()
        num_pix += (np.abs(cf-img_from)[usefule_cf_mask]>0).sum()

        for j in range(cf.shape[0]):
            cf_0 = cf[j]
            cf_from = img_from[j]
            cf_to = img_to

            cf_to[gpu2np(model.mask.sum(dim=1)<1)] = None
            cf_from[gpu2np(model.mask.sum(dim=1)<1)] = None
            cf_0[gpu2np(model.mask.sum(dim=1)<1)] = None

            fig_all_img.add_trace(go.Heatmap(z=cf_from.reshape(8, 8)[::-1]), 1, j+1)
            fig_all_img.add_trace(go.Heatmap(z=cf_to.reshape(8, 8)[::-1]), 2, j+1)
            fig_all_img.add_trace(go.Heatmap(z=np.abs(cf_from-cf_0).reshape(8, 8)[::-1]), 3, j+1)
            fig_all_img.add_trace(go.Heatmap(z=cf_0.reshape(8, 8)[::-1]), 4, j+1)

        log_dict_cf={
            'cm{}/fig_all_img'.format(cf_num): fig_all_img,
            # 'cm{}/fig_cf'.format(cf_num): fig_cf,
        }
        wandb.log(log_dict_cf)
    
    wandb.log({'num_succ_pic':num_succ_pic, 'num_pix':num_pix/num_succ_pic})



if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description="*** author")
    parser.add_argument("--offline", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1, metavar="S")
    parser.add_argument("--data_path", type=str, default="/zangzelin/data")
    parser.add_argument("--log_interval", type=int, default=300)
    parser.add_argument("--project_name", type=str, default="case_study")
    parser.add_argument(
        "--computer", type=str, default=os.popen(
            "git config user.name").read()[:-1]
    )

    # data set param
    parser.add_argument(
        "--data_name",
        type=str,
        default="InsEmb_Digit",
        choices=[
            "OTU",
            "Activity",
            "Gast10k1457",
            "InsEmb_Car2",
            "PBMCD2638",
            "InsEmb_Univ",
            "InsEmb_PBMC",
            "PBMC",
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
    parser.add_argument("--l2alpha", type=float, default=10)
    parser.add_argument("--nu", type=float, default=5e-3)
    parser.add_argument("--num_link_aim", type=float, default=0.2)
    # parser.add_argument("--num_fea_aim", type=int, default=128)
    parser.add_argument("--num_fea_aim", type=int, default=36)
    parser.add_argument("--K_plot", type=int, default=40)

    parser.add_argument("--num_fea_per_pat", type=int, default=10)  # 0.5
    # parser.add_argument("--K", type=int, default=3)
    parser.add_argument("--K", type=int, default=5)
    parser.add_argument("--Uniform_t", type=float, default=-1)  # 0.3
    parser.add_argument("--Bernoulli_t", type=float, default=0.4)  # 0.4
    parser.add_argument("--Normal_t", type=float, default=-1)  # 0.5
    parser.add_argument("--beta", type=float, default=0.1)  
    parser.add_argument("--kappa", type=float, default=0.0)  
    parser.add_argument("--theta", type=float, default=100.0)  

    # train param
    parser.add_argument(
        "--NetworkStructure_1", type=list, default=[-1, 200] + [200] * 5
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
