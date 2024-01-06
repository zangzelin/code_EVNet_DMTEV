import numpy as np
# import torch
# import pandas as pd
# import dice_ml
# import plotly.express as px
# import wandb
# import plotly.graph_objects as go
from alibi.explainers import CounterfactualProto
import tensorflow as tf
# import os
# from alibi.models.tensorflow.cfrl_models import MNISTClassifier
# from alibi.explainers import CounterfactualRL
# import keras
# from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D, Input, UpSampling2D
# from tensorflow.keras.models import Model, load_model
# from alibi.models.tensorflow.autoencoder import AE
# import tensorflow.keras as keras

# from patemb_main import plot_cf_figure

def gpu2np(a):
    return a.cpu().detach().numpy()
    

def SwichExplainability(
    model,
    data_use,
    label_pesodu,
    cluster_number=10,
    cf_example_number=3,
    beta=0.1,
    kappa=0.0,
    theta=100.0,
    cf_max_iterations=1000,
    # k=15,
    k=5,
    pix=0,
    top_cluster=2,
):

    # img_from_list = [[] for i in range(cluster_number)]*cluster_number
    # cf_list = [[] for i in range(cluster_number)]*cluster_number
    img_from_list = []
    cf_list = []

    cluster_list = []
    for i in range(cluster_number):
        if (label_pesodu == i).sum()>15:
            cluster_list.append(i)

    cluster_list_ = cluster_list[:top_cluster]
    # cluster_list_ = cluster_list

    for c_from in cluster_list_:
        img_list_c = []
        cf_list_c = []
        for c_to in cluster_list_:
            if c_from != c_to:
                cf_index_from = np.where(
                    label_pesodu==c_from
                    )[0][:cf_example_number]
                img_from, cf = CF_fromi_to_j(
                        model=model,
                        data_use=data_use,
                        cf_index_from=cf_index_from,
                        cf_index_to=c_to,
                        beta=beta,
                        kappa=kappa,
                        theta=theta,
                        cf_max_iterations=cf_max_iterations,
                        k=k,
                    )
                img_list_c.append(img_from)
                cf_list_c.append(cf)
            else:
                img_list_c.append(None)
                cf_list_c.append(None)
                
        img_from_list.append(img_list_c)
        cf_list.append(cf_list_c)

    return img_from_list, cf_list

def CF_fromi_to_j(
    model,
    data_use,
    cf_index_to=0,
    cf_index_from=[1,2,4,5,6,7,8,12,16,19],
    beta=0.1,
    kappa=0.0,
    theta=100.0,
    cf_max_iterations=5000,
    k=15,
):

    img_from = data_use[cf_index_from]
    cf_data = GenCF_case_stydy_alibi_Proto(
        model_test=model, 
        cf_from=img_from,
        cf_aim=cf_index_to,
        beta=beta,
        kappa=kappa,
        theta=theta,
        cf_max_iterations=cf_max_iterations,
        k=k,
        )

    return img_from, cf_data

def GenCF_case_stydy_alibi_Proto(
    model_test, 
    cf_from, 
    cf_aim=0, 
    num_data=10000,
    beta=0.1,
    kappa=0.0,
    theta=100.,
    c_init=1.,
    c_steps=2,
    cf_max_iterations=2000,
    k=15,
    k_type='point',
    ):
    
    x_train = model_test.data_train.data
    x_train = gpu2np(x_train.reshape((model_test.data_train.data.shape[0],-1)))
    model_test.forward_save = model_test.forward

    mask = gpu2np(model_test.mask>0)
    x_train_feature_mask = x_train[:, mask]
    
    shape = (1,) + x_train_feature_mask.shape[1:]
    tf.compat.v1.disable_v2_behavior()
    model_test.forward = model_test.predict_lime_g_for_numpy
    explainer = CounterfactualProto(
        model_test, shape,
        use_kdtree=True, 
        theta=theta,
        c_init=c_init,
        c_steps=c_steps,
        beta=beta,
        max_iterations=cf_max_iterations,
        feature_range=(x_train_feature_mask.min(), x_train_feature_mask.max())
        )

    explainer.fit(x_train_feature_mask[:num_data])
    exp_r = []
    for i in range(cf_from.shape[0]):
        model_test.aim_cluster = cf_aim
        explanation = explainer.explain(
            cf_from[i:i+1][:, mask],
            target_class=[cf_aim], 
            k=k,
            k_type=k_type, 
            verbose=False,
            )
        try:
            exp_r.append(explanation.cf['X'])
        except:
            exp_r.append(np.zeros_like(cf_from[i:i+1][:, mask]))

    exp_r = np.concatenate(exp_r)

    try:
        exp_base = np.zeros(shape=(exp_r.shape[0], x_train.shape[1]))
        exp_base[:,mask] = exp_r
        return exp_base
    except:
        exp_base = np.zeros(shape=(1, x_train.shape[1]))
        return exp_base