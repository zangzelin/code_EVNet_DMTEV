from dataloader import data_base
import umap
# from sklearn.manifold import TSNE
from openTSNE import TSNE
import plotly.express as px
import wandb
# import eval.eval_core as ec
import eval.eval_core_base as ecb
# import local_exp
import plotly.graph_objects as go
from sklearn.decomposition import PCA
import pacmap
import numpy as np
# from umap.parametric_umap import ParametricUMAP
from sklearn.preprocessing import MinMaxScaler
# from ivis import Ivis
# import pytorch_lightning as pl
# import patemb_main_imagenet
import uuid
import argparse
import torch

def up_mainfig_emb(
    ins_emb, label, scatter_size=3, data_name='', method='',
):
    color = np.array(label)

    Curve = ins_emb[:, 0]
    Curve2 = ins_emb[:, 1]

    ml_mx = max(Curve)
    ml_mn = min(Curve)
    ap_mx = max(Curve2)
    ap_mn = min(Curve2)

    if ml_mx > ap_mx:
        mx = ml_mx
    else:
        mx = ap_mx

    if ml_mn < ap_mn:
        mn = ml_mn
    else:
        mn = ap_mn

    mx = mx + mx * 0.2
    mn = mn - mn * 0.2

    layout = go.Layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        width=1000,
        height=1000,
        autosize=False,
    )

    print('start----------')
    fig = go.Figure(layout=layout)
    # color_set_list = list(set(color.tolist()))
    # for c in color_set_list:
    #     m = color == c

    color_dict = list(px.colors.qualitative.Light24) * 10
    color_list = [color_dict[c % 24] for c in color.tolist()]

    fig.add_trace(
        go.Scatter(
            mode="markers",
            x=ins_emb[:, 0],
            y=ins_emb[:, 1],
            marker_line_width=0,
            # name=c,
            marker=dict(
                size=[scatter_size] * ins_emb.shape[0],
                color=color_list,
                # colorscale='rainbow',
            )
        )
    )
    wandb.log({'fig': fig})
    # if 'IMAGENETBYOLTEST' == data_name or 'IMAGENETBYOLTRAIN' == data_name or 'Mnist' == data_name:
    img_path = "baseline_img/baseline{}_{}_epoch{}_{}.png".format(
        method,
        data_name,
        0,
        str(uuid.uuid1())[:10]
    )
    fig.write_image(img_path, scale=3)
    wandb.save(img_path)
    print('emd----------')

    return fig


def main(
    dataname="IMAGENETBYOL",
    data_path="/zangzelin/data",
    method="Pacmap",
    p1_index=None,
    p2_index=None,
    verbose=False,
):

    wandb.init(
        name="base_line" + dataname + "_" + method,
        project="PatEmb_baseline",
        entity="zangzelin",
        # mode="offline" if bool(args.offline) else "online",
        save_code=True,
        # config=args,
    )

    dataset_f = getattr(data_base, dataname + "Dataset")
    data_train = dataset_f(
        data_name=dataname,
        train=True,
        datapath=data_path,
        # preprocess_bool=False,
    )

    # data = PCA(n_components=300).fit_transform(data_train.data)
    data = data_train.data
    label = data_train.label
    # norm the data
    # scaler = MinMaxScaler()
    # data = scaler.fit_transform(data)

    if method == "umap":
        p1_list = [10, 15, 20, 25]
        p2_list = [0.01, 0.05, 0.08, 0.1, 0.15]
        if p1_index is not None:
            p1 = p1_list[p1_index]
            p2 = p2_list[p2_index]
            reducer_train = umap.UMAP(random_state=0, n_neighbors=int(p1), min_dist=p2, verbose=verbose)
        else:
            reducer_train = umap.UMAP(random_state=0, verbose=verbose)
        ins_emb = reducer_train.fit_transform(X=data)
    if method == "tsne":
        p1_list = [20, 25, 30, 35]
        p2_list = [8, 10, 12, 14, 16]
        if p1_index is not None:
            p1 = p1_list[p1_index]
            p2 = p2_list[p2_index]
            transformer = TSNE(perplexity=int(p1), early_exaggeration=p2, verbose=verbose)
        else:
            transformer = TSNE(verbose=verbose)
        transformer_forward = transformer.fit(data)
        ins_emb = transformer_forward.transform(data)
    if method == "pumap":
        p1_list = [10, 15, 20, 25]
        p2_list = [0.01, 0.05, 0.08, 0.1, 0.15]
        if p1_index is not None:
            p1 = p1_list[p1_index]
            p2 = p2_list[p2_index]
            reducer_train = ParametricUMAP(random_state=0, n_neighbors=int(p1), min_dist=p2, verbose=verbose)
        else:
            reducer_train = ParametricUMAP(random_state=0, verbose=verbose)
        reducer_train.fit(data)
        ins_emb = reducer_train.transform(data)
    if method == "Pacmap":
        p1_list = [10, 15, 20, 25]
        p2_list = [0.3, 0.4, 0.5, 0.6, 0.7]
        if p1_index is not None:
            p1 = p1_list[p1_index]
            p2 = p2_list[p2_index]
            reducer_train = pacmap.PaCMAP(n_components=2, n_neighbors=p1, MN_ratio=p2, verbose=verbose,)
        else:
            reducer_train = pacmap.PaCMAP(n_components=2, verbose=verbose)
        ins_emb = reducer_train.fit_transform(data)
    if method == "ivis":
        p1_list = [130, 140, 150, 160]
        p2_list = [40, 45, 50, 55, 60]
        X_scaled_train = MinMaxScaler().fit_transform(data)
        if p1_index is not None:
            p1 = p1_list[p1_index]
            p2 = p2_list[p2_index]
            reducer_train = Ivis(embedding_dims=2, k=int(p1), ntrees=int(p2))
        else:
            reducer_train = Ivis(embedding_dims=2)
        ins_emb = reducer_train.fit_transform(X_scaled_train)
    # if method == "ours":
    #     ins_emb = np.load("save_checkpoint_use/last_mnist_ins_emb.npy")


    if args.vis_down_sample < ins_emb.shape[0]:
        ins_emb = ins_emb[:args.vis_down_sample]
        label = label[:args.vis_down_sample]

    label = label.detach().cpu().numpy()
    data = data.detach().cpu().numpy()
    e_train = ecb.Eval(input=data, latent=ins_emb, label=label, k=10)
    trai_svc = e_train.E_Classifacation_SVC()
    wandb.log({'final_metric/trai_svc': trai_svc})

    up_mainfig_emb(
        ins_emb, label,
        scatter_size=3 if data.shape[0] > 10000 else 7,
        data_name=dataname,
        method=method,
        )


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description="*** author")
    parser.add_argument(
        "--method",
        type=str,
        default="umap",
        choices=["Pacmap", "umap", "tsne", "pumap", "ivis", "ours"],
    )
    parser.add_argument("--vis_down_sample", type=int, default=200*1000,)
    parser.add_argument("--p1_index", type=int, default=None,)
    parser.add_argument("--p2_index", type=int, default=None,)
    parser.add_argument("--verbose", type=bool, default=False,)
    # data set param
    parser.add_argument(
        "--data_name",
        type=str,
        default="Gast10k1457",
        choices=[
            "IMAGENETBYOLTRAIN",
            "IMAGENETBYOLTEST",
            "BreastCancer",
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
            "MnistBIN",
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
            "EMnistBC200k",
            "EMnistBYCLASS",
            "Cifar10",
            "Colon",
            "Gast10k",
            "HCL60K50D",
            "HCL60K3037D",
            "HCL280K50D",
            "HCL280K3037D",
            "HCL3037D",
            "HCL60K",
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
    
    args = parser.parse_args()
    # args = args.parse_args()
    if args.data_name == 'MCA':
        args.data_name = 'MCAD9119'
    if args.data_name == 'HCL3037D':
        args.data_name = 'HCL60K3037D'

    main(
        dataname=args.data_name,
        data_path="/zangzelin/data",
        method=args.method,
        p1_index=args.p1_index,
        p2_index=args.p2_index,
        verbose=args.verbose,
    )
