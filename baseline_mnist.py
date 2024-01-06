from dataloader import data_base
import umap
from sklearn.manifold import TSNE
import plotly.express as px
import wandb
import eval.eval_core as ec
import local_exp
import plotly.graph_objects as go


if __name__ == "__main__":
    dataname = "Mnist"
    data_path = "/zangzelin/data"

    wandb.init(
        name=dataname,
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
        )

    # ins_emb = umap.UMAP().fit_transform(
    #     X=data_train.data
    # )
    ins_emb = TSNE().fit_transform(data_train.data)
    color = data_train.label

    label_pesodu, cluster_centers = local_exp.Keams_clustering(
            ins_emb, n_clusters=10)
    sf1_2 = ec.Plot_subfig_1_2(
        ins_emb=ins_emb,
        label_pesodu=label_pesodu,
        cluster_centers=cluster_centers,
        # shap_values=shap_values,
        )

    color_set = list(set(color.tolist()))
    fig_list = []
    for c in color_set:
        mask = (color == c)

        fig_data = go.Scatter(
            x=ins_emb[mask, 0],
            y=ins_emb[mask, 1],
            mode='markers',
            marker_line_width=0,
            marker=dict(
                size=[5]*color.shape[0],
                # color=color[mask].astype(np.int32),
                ),
            )
        fig_list.append(fig_data)

    fig1 = go.Figure(data=fig_list)
    fig2 = go.Figure(data=sf1_2)
    # fig = px.scatter(
    #         x=ins_emb[:, 0], y=ins_emb[:, 1],
    #         color=[str(c) for c in color])
    wandb.log({'fig': fig1, 'fig2': fig2})
