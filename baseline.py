import os
import sys
import pandas as pd
import numpy as np
import sklearn
# import torch
from PIL import Image
from sklearn.datasets import fetch_openml, make_s_curve, make_swiss_roll
from skfeature.function.sparse_learning_based import UDFS
from skfeature.function.sparse_learning_based import MCFS
from skfeature.function.sparse_learning_based import NDFS
from skfeature.function.similarity_based import lap_score
from skfeature.utility.sparse_learning import feature_ranking
from skfeature.utility import construct_W
from baseline.Concrete_Autoencoders.experiments.generate_comparison_figures import aefs_subset_selector as aefs_subset_selector
from baseline.FAE.fae import FAEFS
from baseline.Concrete_Autoencoders.experiments.concrete_estimator import run_experiment as run_experiment
from baseline import ivfs
# from baseline.Concrete_Autoencoders.experiments.generate_comparison_figures import pfa_selector as pfa_selector
# import random
# from sklearn.model_selection import train_test_split
# import tensorflow as tf
import wandb
# import umap
# import load_data_f.dataset as datasetfunc
import pytorch_lightning as pl
from dataloader import data_base
import torch
# import load_disF.disfunc as disfunc
# import load_simF.simfunc as simfunc
import eval.eval_core as ec

# import plotly.express as px
# from sklearn.manifold import TSNE
# from sklearn.decomposition import PCA
# # from umap.parametric_umap import ParametricUMAP
# # from ivis import Ivis
# from sklearn.preprocessing import MinMaxScaler
# # import phate
# import matplotlib.pyplot as plt


def showMaskHeatMap(mask):
    fig, ax = plt.subplots(figsize=(5, 5))
    data = mask  #.reshape(dim_i, dim_j)
    N_allF = len(mask)
    N_c = int(np.sqrt(N_allF))
    N_r = N_allF // N_c
    if N_c * N_r < N_allF:
        N_r += 1
    data = np.concatenate([data, np.array([0] * (N_c * N_r - N_allF))
                           ]).reshape(N_c, N_r)
    data[data < 0.5] = 0
    im = plt.imshow(data)
    plt.colorbar(im)
    # for i in range(dim_i):
    #     for j in range(dim_j):
    #         text = ax.text(j, i, str(data[i, j])[:4],
    #                     ha="center", va="center", color="w")
    return fig


def get_one_hot(targets, nb_classes):
    a = np.array(targets)
    res = np.eye(int(nb_classes))[a.astype(np.int32).reshape(-1)]
    a = res.reshape(list(targets.shape) + [int(nb_classes)])
    if len(a.shape) > 2:
        a = a[:, 0, :]
    return a


# import tool
if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(description='*** author')
    parser.add_argument(
        '--name',
        type=str,
        default='Mnist',
    )
    parser.add_argument('--dataname',
                        type=str,
                        default='arcene',
                        choices=[
                            'Mnist',
                            'KMnist',
                            'EMnistBC',
                            'arcene',
                            'Gast10k1457',
                            'MCAD9119',
                            'HCL60K3037D',
                        ])
    parser.add_argument(
        '--n_point',
        type=int,
        default=600000000,
    )
    parser.add_argument('--num_fea', type=int, default=64)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--project_name', type=str, default='')
    parser.add_argument('--method',
                        type=str,
                        default='ivfs',
                        choices=[
                            'UDFS', 'MCFS', 'NDFS',
                            'lap_score', 'pfa',
                            'aefs', 'CAE', 'FAE', 'ivfs'
                        ])
    parser.add_argument("--data_path", type=str, default="/root/data")
    args = pl.Trainer.add_argparse_args(parser)
    args = args.parse_args()

    info = [str(s) for s in sys.argv[1:]]
    runname = "_".join(["baseline", args.dataname, "".join(info)])

    wandb.init(
        name=runname,
        project='OTN_baseline_' + args.__dict__['project_name'],
        entity='zangzelin',
        config=args,
    )

    dataset_f = getattr(data_base, args.dataname + "Dataset")
    dataset_train = dataset_f(
        data_name=args.dataname,
        train=True,
        datapath=args.data_path,
    )
    dataset_train.to_device('cpu')
    dataset_test = dataset_f(
        data_name=args.dataname,
        train=False,
        datapath=args.data_path,
    )
    dataset_test.to_device('cpu')

    data = dataset_train.data.detach().cpu().numpy()
    data_test = dataset_test.data.detach().cpu().numpy()
    label = dataset_train.label.detach().cpu().numpy()
    label_test = dataset_test.label.detach().cpu().numpy()

    if args.__dict__['num_fea'] < 0:
        num_fea_dict = {
            'MiceProtein': 8,
            'Coil20': 64,
            'Activity': 64,
            'Mnist': 64,
            'FMnist': 64,
            'KMnist': 64,
            'EMnistBC': 64,
            'Gast10k1457': 64,
            'GLIOMA': 64,
            'leukemia': 64,
            'pixraw10P': 64,
            'Prostatege': 64,
            'arcene': 64,
            'HCL280K3037D': 64,
            'Gast10k1457': 64,
            'MCA': 64,
            'MCAD9119': 64,
            'PBMCD2638': 64,
            'HCL60K3037D': 64,
            # 'HCL280K3037D':,
        }
        num_fea = num_fea_dict[args.__dict__['dataname']]
    else:
        num_fea = min(args.__dict__['num_fea'], data.shape[1])

    pl.utilities.seed.seed_everything(seed=args.__dict__['seed'])

    if args.__dict__['method'] == 'UDFS':
        kwargs = {
            'k': 5,
        }
        Weight = UDFS.udfs(data, gamma=0.1, k=np.min([num_fea, data.shape[0]]))
        idx = feature_ranking(Weight)
    if args.__dict__['method'] == 'MCFS':
        Weight = MCFS.mcfs(data, n_selected_features=num_fea)
        idx = feature_ranking(Weight)
    if args.__dict__['method'] == 'NDFS':
        Weight = NDFS.ndfs(data, n_clusters=np.min([num_fea, data.shape[0]]))
        idx = feature_ranking(Weight)
    if args.__dict__['method'] == 'lap_score':
        kwargs_W = {
            "metric": "euclidean",
            "neighbor_mode": "knn",
            "weight_mode": "heat_kernel",
            "k": 5,
            't': 1
        }
        W = construct_W.construct_W(data, **kwargs_W)
        score = lap_score.lap_score(data, W=W)
        idx = lap_score.feature_ranking(score)
    if args.__dict__['method'] == 'aefs':
        idx, _ = aefs_subset_selector(
            [data, get_one_hot(label, np.max(label) + 1)],
            K=np.min([num_fea, data.shape[0]]))
    if args.__dict__['method'] == 'pfa':
        idx = pfa_selector(data, num_fea)
    if args.__dict__['method'] == 'CAE':
        data_input = [data, get_one_hot(label, np.max(label) + 1)]
        probabilities = run_experiment(
            'zzl',
            data_input,
            data_input,
            data_input,
            num_fea,
            [],
            num_epochs=100,
            batch_size=256,
            learning_rate=0.01,
            dropout=0.0,
        )
        idx = np.argmax(probabilities, axis=1)
    if args.__dict__['method'] == 'FAE':
        print('use FAE method ')
        probabilities = FAEFS(data, data, data, num_fea, 500)
        print(probabilities.shape)
        idx = np.argsort(probabilities)
    if args.__dict__['method'] == 'ivfs':
        print('use ivfs method ')
        probabilities = ivfs.ivfs_selector(
            data,
            tilde_feature=num_fea,
            tilde_sample=data.shape[0] // 10,
            k=1000,
        )
        idx = np.argsort(probabilities)

    # idx = feature_ranking(Weight)
    selected_features = data_test[:, idx[0:num_fea]]
    mask = np.zeros((data.shape[1]))
    mask[idx[0:num_fea]] = 1

    e = ec.Eval(
        input=data_test,
        latent=data_test,
        label=label_test,
        # ------------------------------------------
        train_input=data,
        train_latent=data,
        train_label=label,
        # ------------------------------------------
        mask=mask > 0.5,
    )

    wandb_logs = {}
    N_Feature = np.sum(mask > 0.5)
    wandb_logs.update({
        "metric/#Feature":
        N_Feature,
        "n_rate":
        e.GraphMatch(),
        "vis/Mask":
        ec.showMask(torch.tensor(mask)),
        "vis/VisSelectUMAP":
        e.VisSelectUMAP(data, label),
        "vis/VisAllUMAP":
        e.VisAllUMAP(data, label),
        "selected_index":
        ",".join([
            str(a) for a in torch.tensor(mask).detach().sort()[1][(
                -1 * num_fea):].cpu().numpy().tolist()
        ]),
    })

    ec.Test_ET_CV(e, wandb_logs, 0)
    wandb_logs.update(
        ec.ShowEmb(data, dataset_train.labelstr,
                   np.array(range(data.shape[0]))))

    wandb.log(wandb_logs)