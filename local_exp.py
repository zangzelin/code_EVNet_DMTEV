from sklearn.cluster import KMeans
import shap
from symtable import Symbol
import uuid
import scipy

# import sklearn
import torch
from sklearn.cluster import SpectralClustering
import sklearn
from sklearn.preprocessing import MinMaxScaler
import scipy.spatial as spt

# from scipy.spatial.distance import squareform
# from scipy.stats import spearmanr
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn import metrics
from sklearn.svm import SVC
import numpy as np
import plotly.figure_factory as ff

# from transformers import RagRetriever
import wandb

# import random
from sklearn.metrics import pairwise_distances

from sklearn.linear_model import LinearRegression

# from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from munkres import Munkres
from sklearn.neighbors import kneighbors_graph
import matplotlib.pyplot as plt
import umap
import plotly.express as px
from sklearn.ensemble import ExtraTreesClassifier
import plotly.graph_objects as go


def gpu2np(a):
    return a.cpu().detach().numpy()


def feag_cluster_mask(
    shap_values, mask, fea_cluster_centers, fea_label_pesodu, n_feverycluset
):
    shap_values_fea_pat = shap_values.mean(axis=1).T
    shap_values_fea_pat = shap_values_fea_pat[gpu2np(mask)]
    shap_values_fea_group = []
    for i in range(fea_cluster_centers.shape[0]):
        shap_values_fea_group.append(
            shap_values_fea_pat[fea_label_pesodu == i].mean(axis=0, keepdims=1)
        )
    shap_values_fea_group = np.concatenate(shap_values_fea_group)
    shap_values_fea_group_topk = np.sort(shap_values_fea_group, axis=0)[::-1][
        n_feverycluset
    ]
    mask_fea_pat = shap_values_fea_group > shap_values_fea_group_topk
    return mask_fea_pat, shap_values_fea_group


def Keams_clustering(ins_emb, n_clusters):
    KMeans_model = KMeans(
        n_clusters=n_clusters,
        n_jobs=None,
    ).fit(ins_emb)
    label_pesodu = KMeans_model.labels_
    cluster_centers = KMeans_model.cluster_centers_
    return label_pesodu, cluster_centers


def LocalExplainability(
    data,
    model,
    num_s_shap=5,
):

    model.forward = model.predict_lime_g
    explainer = shap.GradientExplainer(
    # explainer = shap.DeepExplainer(
        model,
        data.to(model.mask.device),
    )

    shap_values = explainer.shap_values(
        data.to(model.mask.device)[0:num_s_shap]
    )
    shap_values_abs = np.abs(np.array(shap_values))

    shap_values_fea_ins = shap_values_abs.mean(axis=0).T[gpu2np(model.mask)]
    fea_most_import_ins_index = np.argsort(
        shap_values_fea_ins,
        axis=1,
    )[:, -2:]
    fake_ins_for_fea = [
        data[fea_most_import_ins_index[i][0:1]] * 0.2
        + data[fea_most_import_ins_index[i][1:2]] * 0.8
        for i in range(fea_most_import_ins_index.shape[0])
    ]
    fake_ins_for_fea = np.concatenate(fake_ins_for_fea)

    fea_emb = model.forward_fea(torch.tensor(fake_ins_for_fea))[2]
    fea_emb = gpu2np(fea_emb)

    return shap_values_abs, fea_emb, shap_values
