# from cv2 import exp
import numpy as np
import torch
import pandas as pd
# import dice_ml
import plotly.express as px
import wandb
import plotly.graph_objects as go
from alibi.explainers import CounterfactualProto
import tensorflow as tf
import os
from alibi.models.tensorflow.cfrl_models import MNISTClassifier
from alibi.explainers import CounterfactualRL
import keras
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D, Input, UpSampling2D
from tensorflow.keras.models import Model, load_model
from alibi.models.tensorflow.autoencoder import AE
import tensorflow.keras as keras
# tf.compat.v1.disable_v2_behavior()# 关键
# tf.compat.v1.disable_eager_execution()

class MNISTEncoder(keras.Model):
    """
    MNIST encoder used in the experiments for the Counterfactual with Reinforcement Learning. The model
    consists of 3 convolutional layers having 16, 8 and 8 channels and a kernel size of 3, with ReLU nonlinearities.
    Each convolutional layer is followed by a maxpooling layer of size 2. Finally, a fully connected layer
    follows the convolutional block with a tanh nonlinearity. The tanh clips the output between [-1, 1], required
    in the DDPG algorithm (e.g., [act_low, act_high]). The embedding dimension used in the paper is 32, although
    this can vary.
    """

    def __init__(self, latent_dim: int, **kwargs) -> None:
        """
        Constructor.

        Parameters
        ----------
        latent_dim
            Latent dimension.
        """
        super().__init__(**kwargs)

        # self.conv1 = keras.layers.Conv2D(16, 3, padding="same", activation="relu")
        # self.maxpool1 = keras.layers.MaxPool2D(pool_size=(2, 2), strides=2)
        # self.conv2 = keras.layers.Conv2D(8, 3, padding="same", activation="relu")
        # self.maxpool2 = keras.layers.MaxPool2D(pool_size=(2, 2), strides=2)
        # self.conv3 = keras.layers.Conv2D(8, 3, padding="same", activation="relu")
        # self.maxpool3 = keras.layers.MaxPool2D(pool_size=(2, 2), strides=2)
        # self.flatten = keras.layers.Flatten()
        self.fc1 = keras.layers.Dense(1000, activation='tanh')
        self.fc2 = keras.layers.Dense(1000, activation='tanh')
        self.fc3 = keras.layers.Dense(latent_dim, activation='tanh')
        # self.fc1 = keras.layers.Dense(latent_dim, activation='tanh')
        # self.fc1 = keras.layers.Dense(latent_dim, activation='tanh')

    def call(self, x: tf.Tensor, **kwargs) -> tf.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x
            Input tensor.
        **kwargs
            Other arguments. Not used.

        Returns
        -------
        Encoding representation having each component in the interval [-1, 1]
        """
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


class MNISTDecoder(keras.Model):
    """
    MNIST decoder used in the Counterfactual with Reinforcement Learning experiments. The model consists of a fully
    connected layer of 128 units with ReLU activation followed by a convolutional block. The convolutional block
    consists fo 4 convolutional layers having 8, 8, 8  and 1 channels and a kernel size of 3. Each convolutional layer,
    except the last one, has ReLU nonlinearities and is followed by an up-sampling layer of size 2. The final layers
    uses a sigmoid activation to clip the output values in [0, 1].
    """

    def __init__(self, out_dim: int, **kwargs) -> None:
        """ Constructor. """
        super().__init__(**kwargs)

        self.fc1 = keras.layers.Dense(1000, activation='tanh')
        self.fc2 = keras.layers.Dense(1000, activation='tanh')
        self.fc3 = keras.layers.Dense(out_dim, activation='sigmoid')

    def call(self, x: tf.Tensor, **kwargs) -> tf.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x
            Input tensor
        **kwargs
            Other arguments. Not used.

        Returns
        -------
        Decoded input having each component in the interval [0, 1].
        """
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x





# tf.compat.v1.disable_v2_behavior() 
def gpu2np(a):
    return a.cpu().detach().numpy()


def GenCF(model, cf_index_from=500, n_cf=4, cf_aim=[0, 0], sample_size=1000):

    # model.eval()
    model_test = model
    # model_test.m = model_test.m.to('cpu')
    # model_test.mask = model_test.mask.to('cpu')
    label = np.zeros((model_test.data_train.data.shape[0], 1)).astype(np.int32)
    best_index = torch.topk(
        (
            model_test.forward(
            model_test.data_train.data
            )[2]-torch.tensor(cf_aim).to(model_test.data_train.data.device)
        ).norm(dim=1),
        k=10, largest=False,
    )[1]
    model_test.cf_aim = cf_aim
    # label[] = 1
    label[gpu2np(best_index)] = 1

    dataframe = pd.DataFrame(
        np.concatenate([gpu2np(model_test.data_train.data), label], axis=1),
        columns=[str(i) for i in range(1 + model_test.data_train.data[0:1].shape[1])],
    )
    out_feature_index = model_test.data_train.data.shape[1]
    feature_name = [n for n in dataframe.columns if n != str(out_feature_index)]
    d = dice_ml.Data(
        dataframe=dataframe, 
        continuous_features=feature_name, 
        outcome_name=str(out_feature_index)
    )
    # feature_name = [n for n in dataframe.columns if n != "50"]
    # d = dice_ml.Data(
    #     dataframe=dataframe, continuous_features=feature_name, outcome_name="50"
    # )
    m = dice_ml.Model(
        model=model_test, backend="sklearn", model_type="regressor"
    ) 
    exp = dice_ml.Dice(d, m, method="random")
    # exp = dice_ml.Dice(d, m, method='genetic')

    sample = pd.DataFrame(
        gpu2np(model_test.data_train.data[cf_index_from : cf_index_from + 1]),
        columns=feature_name,
    )
    dice_exp = exp.generate_counterfactuals(
        sample,
        total_CFs=n_cf,
        sample_size=sample_size,
        posthoc_sparsity_param=0.9,
        desired_range=[0.85, 0.99],
        verbose=True,
    )

    return dice_exp.cf_examples_list[0].final_cfs_df


def GenCF_case_stydy(
    model, 
    cf_index_from=500, 
    n_cf=4, 
    cf_aim=[0, 0], 
    sample_size=1000,
    features_to_vary='all',
    ):

    # model.eval()
    model_test = model
    # model_test.m = model_test.m.to('cpu')
    # model_test.mask = model_test.mask.to('cpu')
    label = np.zeros((model_test.data_train.data.shape[0], 1)).astype(np.int32)
    best_index = torch.topk(
        (
            model_test.forward(
            model_test.data_train.data
            )[2]-torch.tensor(cf_aim).to(model_test.data_train.data.device)
        ).norm(dim=1),
        k=10, largest=False,
    )[1]
    model_test.cf_aim = cf_aim
    # label[] = 1
    label[gpu2np(best_index)] = 1

    dataframe = pd.DataFrame(
        np.concatenate([gpu2np(model_test.data_train.data), label], axis=1),
        columns=[str(i) for i in range(1 + model_test.data_train.data[0:1].shape[1])],
    )
    out_feature_index = model_test.data_train.data.shape[1]
    feature_name = [n for n in dataframe.columns if n != str(out_feature_index)]
    d = dice_ml.Data(
        dataframe=dataframe, 
        continuous_features=feature_name, 
        outcome_name=str(out_feature_index)
    )
    # feature_name = [n for n in dataframe.columns if n != "50"]
    # d = dice_ml.Data(
    #     dataframe=dataframe, continuous_features=feature_name, outcome_name="50"
    # )
    m = dice_ml.Model(
        model=model_test, backend="sklearn", model_type="regressor",
    ) 
    exp = dice_ml.Dice(d, m, method="random")
    # exp = dice_ml.Dice(d, m, method='genetic')

    sample = pd.DataFrame(
        gpu2np(model_test.data_train.data[cf_index_from : cf_index_from + 1]),
        columns=feature_name,
    )
    dice_exp = exp.generate_counterfactuals(
        sample,
        total_CFs=n_cf,
        sample_size=sample_size,
        posthoc_sparsity_param=0.1,
        features_to_vary=features_to_vary,
        # posthoc_sparsity_algorithm="binary",
        desired_range=[0.8, 0.99],
        verbose=True,
    )

    return dice_exp.cf_examples_list[0].final_cfs_df



def CF(
    model,
    cf_index_to=0,
    cf_index_from=500,
    n_cf=4,
    cf_aim=[0, 0],
    fig_main=None,
    fig_main_row=2,
    fig_main_col=3,
):

    datashow_mid = gpu2np(model(model.data_train.data)[2])
    dict_cf = {}
    cf_data = GenCF(model=model, cf_index_from=cf_index_from, n_cf=n_cf, cf_aim=datashow_mid[cf_index_to])
    if isinstance(cf_data, pd.DataFrame):
        datashow_data = np.concatenate(
            [cf_data.to_numpy()[:, :-1], gpu2np(model.data_train.data)]
        )
        datashow_label = np.array(
            [2] * n_cf + [0] * gpu2np(model.data_train.data).shape[0]
        )
        datashow_label[cf_index_from] = 1
        datashow_mid = gpu2np(model(torch.tensor(datashow_data).to("cuda"))[2])

        # fig = px.scatter(
        #     x=datashow_mid[:, 0], y=datashow_mid[:, 1], color=datashow_label
        # )

        # cf_table = []
        for cf in range(n_cf):
            data_show_line = datashow_mid[[n_cf + cf_index_from, cf], :]
            a = gpu2np(model.data_train.data)[cf_index_from]
            b = cf_data.to_numpy()[:, :-1][cf]
            str_use = []
            feature_use_bool = gpu2np(model.mask).sum(axis=1) > 0
            bool_eq = [False] * (cf_data.shape[1] - 1)
            o_list = []
            cf_list = []
            for i in range(cf_data.shape[1] - 1):
                if a[i] != b[i] and feature_use_bool[i]:
                    str_use.append("f{}".format(i))
                    bool_eq[i] = True
                    o_list.append(a[i])
                    cf_list.append(b[i])
            # fig.add_trace(
            #     go.Scatter(
            #         x=data_show_line[:, 0],
            #         y=data_show_line[:, 1],
            #     )
            # )

            cf_data_show = pd.DataFrame(
                np.concatenate(
                    [a.reshape(-1)[bool_eq], b.reshape(-1)[bool_eq]], axis=0
                ),
                index=str_use + str_use,
                columns=["v"],
            )
            cf_data_show["type"] = np.array(
                ["origin"] * np.sum(bool_eq) + ["cf"] * np.sum(bool_eq)
            )
            cf_data_show["findex"] = str_use + str_use

            dict_cf["cf/cf_{}".format(cf)] = px.histogram(
                cf_data_show, barmode="group", x="findex", y="v", color="type"
            )

        # dict_cf["cf/cf"] = fig
        # wandb.log(dict_cf)

        fig_main.add_trace(
            go.Scatter(
                mode="markers",
                name="",
                x=datashow_mid[:, 0],
                y=datashow_mid[:, 1],
                marker_line_width=0,
                marker=dict(
                    size=[5] * datashow_label.shape[0],
                    color=datashow_label,
                )
                # color=datashow_label
            ),
            row=1,
            col=3,
        )
        # fig_main.add_trace(
        #     go.Scatter(
        #         x=data_show_line[:, 0],
        #         y=data_show_line[:, 1],
        #     ),
        #     row=1,
        #     col=3,
        # )
        fig_main.add_annotation(
            x=data_show_line[1, 0],  # arrows' head
            y=data_show_line[1, 1],  # arrows' head
            # ax=data_show_line[0, 0],  # arrows' tail
            ax=data_show_line[0, 0],  # arrows' tail
            ay=data_show_line[0, 1],  # arrows' tail
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

        # fig_main.add_annotation(
        #     xref="x domain",
        #     yref="y",
        #     x=0.75,
        #     y=1,
        #     text="An annotation whose text and arrowhead reference the axes and the data",
        #     # If axref is exactly the same as xref, then the text's position is
        #     # absolute and specified in the same coordinates as xref.
        #     axref="x domain",
        #     # The same is the case for yref and ayref, but here the coordinates are data
        #     # coordinates
        #     ayref="y",
        #     ax=0.5,
        #     ay=2,
        #     arrowhead=2,
        #     row=1, col=3,
        # )


        fig_main.add_trace(
            go.Bar(
                x=str_use,
                y=o_list,
                # base=[-500,-600,-700],
                marker_color="crimson",
                name="origin_data",
            ),
            row=2,
            col=3,
        )
        fig_main.add_trace(
            go.Bar(
                x=str_use,
                y=cf_list,
                # base=0,
                marker_color="lightslategrey",
                name="cf_data",
            ),
            row=2,
            col=3,
        )

        return fig_main
    return fig_main
        # fig_main.add_trace(
        #     go.Histogram(
        #         cf_data_show, barmode='group',
        #         x='findex', y='v', color='type'
        #         ),
        #     row=2, col=3,
        # )



def CF_case_study(
    model,
    cf_index_to=0,
    cf_index_from=500,
    n_cf=4,
    num_data=5000,
):

    # fig = 

    datashow_mid = gpu2np(model(model.data_train.data[:num_data])[2])
    dict_cf = {}
    cf_data = GenCF_case_stydy(
        model=model, 
        cf_index_from=cf_index_from, 
        n_cf=n_cf, 
        cf_aim=datashow_mid[cf_index_to],
        sample_size=10000,
        )
    if isinstance(cf_data, pd.DataFrame):
        datashow_data = np.concatenate(
            [cf_data.to_numpy()[:, :-1], 
            gpu2np(model.data_train.data[:num_data])]
        )
        datashow_label = np.array(
            [2] * n_cf + [0] * gpu2np(model.data_train.data[:num_data]).shape[0]
        )
        datashow_label[cf_index_from] = 1
        datashow_mid = gpu2np(model(
            torch.tensor(datashow_data).to("cuda"))[2])

        fig = go.Figure(data=[
            go.Scatter(
                x=datashow_mid[:, 0],
                y=datashow_mid[:, 1],
                mode="markers",
                marker_line_width=0,
                marker=dict(
                    size=[5] * datashow_label.shape[0],
                    color=datashow_label,
                )
                )
            ])

        # 
        # px.scatter(
        #     x=datashow_mid[:, 0], y=datashow_mid[:, 1], color=datashow_label
        # )

        # cf_table = []
        for cf in range(n_cf):
            data_show_line = datashow_mid[[n_cf + cf_index_from, cf], :]
            a = gpu2np(model.data_train.data[:num_data])[cf_index_from]
            b = cf_data.to_numpy()[:, :-1][cf]
            str_use = []
            feature_use_bool = gpu2np(model.mask).sum(axis=1) > 0
            bool_eq = [False] * (cf_data.shape[1] - 1)
            o_list = []
            cf_list = []
            for i in range(cf_data.shape[1] - 1):
                if a[i] != b[i] and feature_use_bool[i]:
                    str_use.append("f{}".format(i))
                    bool_eq[i] = True
                    o_list.append(a[i])
                    cf_list.append(b[i])



            cf_data_show = pd.DataFrame(
                np.concatenate(
                    [a.reshape(-1)[bool_eq], 
                    b.reshape(-1)[bool_eq]], axis=0
                ),
                index=str_use + str_use,
                columns=["v"],
            )
            cf_data_show["type"] = np.array(
                ["origin"] * np.sum(bool_eq) + ["cf"] * np.sum(bool_eq)
            )
            cf_data_show["findex"] = str_use + str_use

            dict_cf["cf/cf_{}".format(cf)] = px.histogram(
                cf_data_show, barmode="group", x="findex", y="v", color="type"
            )

        img_to = gpu2np(model.data_train.data[cf_index_to])
        img_from = gpu2np(model.data_train.data[cf_index_from])

        fig.add_annotation(
            x=data_show_line[1, 0],  # arrows' head
            y=data_show_line[1, 1],  # arrows' head
            # ax=data_show_line[0, 0],  # arrows' tail
            ax=data_show_line[0, 0],  # arrows' tail
            ay=data_show_line[0, 1],  # arrows' tail
            xref='x',
            yref='y',
            axref='x',
            ayref='y',
            text='',  # if you want only the arrow
            showarrow=True,
            width=2,
            arrowhead=5,
            arrowsize=5,
            arrowwidth=1,
            arrowcolor='red',
            # row=1, col=3,
          )

        return img_to, img_from, cf_data.to_numpy()[:, :-1], fig



def CF_case_study_abili(
    model,
    cf_index_to=0,
    cf_index_from=500,
    n_cf=1,
    num_data=5000,
    beta=0.1,
    kappa=0.0,
):

    # fig = 

    datashow_mid = gpu2np(model(model.data_train.data[:num_data])[2])
    dict_cf = {}
    cf_data = GenCF_case_stydy_alibi(
        model=model, 
        cf_index_from=cf_index_from, 
        # n_cf=n_cf, 
        cf_aim=datashow_mid[cf_index_to],
        beta=beta,
        kappa=kappa,
        # sample_size=10000,
        )
    if isinstance(cf_data, np.ndarray):
        datashow_data = np.concatenate(
            [cf_data, 
            gpu2np(model.data_train.data[:num_data])]
        )
        datashow_label = np.array(
            [2] * n_cf + [0] * gpu2np(model.data_train.data[:num_data]).shape[0]
        )
        datashow_label[cf_index_from] = 1
        datashow_mid = gpu2np(model(
            torch.tensor(datashow_data).to("cuda"))[2])

        fig = go.Figure(data=[
            go.Scatter(
                x=datashow_mid[:, 0],
                y=datashow_mid[:, 1],
                mode="markers",
                marker_line_width=0,
                marker=dict(
                    size=[5] * datashow_label.shape[0],
                    color=datashow_label,
                )
                )
            ])
        fig.add_trace(
            go.Scatter(
                x=datashow_mid[0:1, 0],
                y=datashow_mid[0:1, 1],
                name='cf',
                mode="markers",
                marker_line_width=0,
                marker=dict(
                    size=[20] * datashow_label.shape[0],
                )
                )
        )        
        fig.add_trace(
            go.Scatter(
                x=datashow_mid[cf_index_to+1:cf_index_to+2, 0],
                y=datashow_mid[cf_index_to+1:cf_index_to+2, 1],
                name='to',
                mode="markers",
                marker_line_width=0,
                marker=dict(
                    size=[20] * datashow_label.shape[0],
                )
                )
        )     
        fig.add_trace(
            go.Scatter(
                x=datashow_mid[cf_index_from+1:cf_index_from+2, 0],
                y=datashow_mid[cf_index_from+1:cf_index_from+2, 1],
                name='from',
                mode="markers",
                marker_line_width=0,
                marker=dict(
                    size=[20] * datashow_label.shape[0],
                )
                )
        )# 
        # px.scatter(
        #     x=datashow_mid[:, 0], y=datashow_mid[:, 1], color=datashow_label
        # )

        # cf_table = []
        # for cf in range(n_cf):
        cf=0
        data_show_line = datashow_mid[[n_cf + cf_index_from, cf], :]
        a = gpu2np(model.data_train.data[:num_data])[cf_index_from]
        b = cf_data[cf]
        str_use = []
        feature_use_bool = gpu2np(model.mask).sum(axis=1) > 0
        bool_eq = [False] * (cf_data.shape[1])
        o_list = []
        cf_list = []
        for i in range(cf_data.shape[0]):
            if a[i] != b[i] and feature_use_bool[i]:
                str_use.append("f{}".format(i))
                bool_eq[i] = True
                o_list.append(a[i])
                cf_list.append(b[i])

        cf_data_show = pd.DataFrame(
            np.concatenate(
                [a.reshape(-1)[bool_eq], 
                b.reshape(-1)[bool_eq]], axis=0
            ),
            index=str_use + str_use,
            columns=["v"],
        )
        cf_data_show["type"] = np.array(
            ["origin"] * np.sum(bool_eq) + ["cf"] * np.sum(bool_eq)
        )
        cf_data_show["findex"] = str_use + str_use

        dict_cf["cf/cf_{}".format(cf)] = px.histogram(
            cf_data_show, barmode="group", x="findex", y="v", color="type"
        )

        img_to = gpu2np(model.data_train.data[cf_index_to])
        img_from = gpu2np(model.data_train.data[cf_index_from])

        fig.add_annotation(
            x=data_show_line[1, 0],  # arrows' head
            y=data_show_line[1, 1],  # arrows' head
            ax=data_show_line[0, 0],  # arrows' tail
            ay=data_show_line[0, 1],  # arrows' tail
            xref='x',
            yref='y',
            axref='x',
            ayref='y',
            text='',  # if you want only the arrow
            showarrow=True,
            width=2,
            arrowhead=5,
            arrowsize=5,
            arrowwidth=1,
            arrowcolor='red',
          )

        return img_to, img_from, cf_data, fig



def GenCF_case_stydy_alibi(
    model, 
    cf_index_from=500, 
    cf_aim=[0,0], 
    # features_to_vary='all',
    num_data=10000,
    beta=0.1,
    kappa=0.0,
    ):

    # model.eval()
    model_test = model
    
    x_train = model_test.data_train.data
    x_train = gpu2np(x_train.reshape((model_test.data_train.data.shape[0],-1)))
    model_test.forward_save = model_test.forward
    model_test.forward = model_test.predict_
    
    X = gpu2np(model_test.data_train.data[cf_index_from : cf_index_from + 1])
    model_test.cf_aim = cf_aim

    # predict_fn = lambda x: model_test.predict_

    mask = gpu2np(model_test.mask>0)
    x_train_feature_mask = x_train[:, mask]
    X_after_mask = X[:, mask]
    
    shape = (1,) + x_train_feature_mask.shape[1:]
    cf = CounterfactualProto(
        model_test, shape,
        use_kdtree=True, 
        kappa=kappa,
        gamma=100.,
        theta=100.,
        c_init=1.,
        c_steps=2,
        beta=beta,
        max_iterations=2000,
        feature_range=(x_train_feature_mask.min(), x_train_feature_mask.max())
        )
    cf.fit(x_train_feature_mask)
    
    explanation = cf.explain(X_after_mask, Y=np.array([[0,1]]),
        target_class=[0], k=15, k_type='point', verbose=False)
 
    model_test.forward = model_test.forward_save

    try:
        exp_base = np.zeros(shape=(1, x_train.shape[1]))
        exp_base[:,mask] = explanation.cf['X']
        return exp_base
    except:
        exp_base = np.zeros(shape=(1, x_train.shape[1]))
        return exp_base


def CF_case_study_abili_RL(
    model,
    cf_index_to=[0],
    cf_index_from=[500, 501],
    n_cf=1,
    num_data=5000,
    beta=0.1,
    kappa=0.0,
):

    # fig = 
    cf_index_from = [1,2,4,5,6,7,8,12,16,19]
    data_use = gpu2np(model.data_train.data[:num_data])
    img_from = data_use[cf_index_from]
    img_to = data_use[cf_index_to]

    datashow_mid = gpu2np(model(model.data_train.data[:num_data])[2])
    # dict_cf = {}
    cf_data = GenCF_case_stydy_alibi_RL(
        model=model, 
        cf_from=gpu2np(model.data_train.data[cf_index_from]),
        cf_aim=datashow_mid[cf_index_to],
        beta=beta,
        kappa=kappa,
        # sample_size=10000,
        )

    return img_to, img_from, cf_data



def GenCF_case_stydy_alibi_RL(
    model, 
    cf_from, 
    cf_aim=[0,0], 
    num_data=10000,
    beta=0.1,
    kappa=0.0,
    ):

    # model.eval()
    model_test = model
    
    x_train = model_test.data_train.data
    x_train = gpu2np(x_train.reshape((model_test.data_train.data.shape[0],-1)))
    model_test.forward_save = model_test.forward
    model_test.forward = model_test.predict_

    # X = gpu2np(model_test.data_train.data[cf_index_from : cf_index_from + 1])
    model_test.cf_aim = cf_aim

    # predict_fn = lambda x: model_test.predict_

    mask = gpu2np(model_test.mask>0)
    x_train_feature_mask = x_train[:, mask]
    # X_after_mask = X[:, mask]
    
    # ------
    BATCH_SIZE = 64
    BUFFER_SIZE = 1024
    trainset_ae = tf.data.Dataset.from_tensor_slices(x_train_feature_mask)
    trainset_ae = trainset_ae.map(lambda x: (x, x))
    trainset_ae = trainset_ae.shuffle(buffer_size=BUFFER_SIZE).batch(BATCH_SIZE)

    # Define autoencode testset.
    testset_ae = tf.data.Dataset.from_tensor_slices(x_train_feature_mask)
    testset_ae = testset_ae.map(lambda x: (x, x))
    testset_ae = testset_ae.shuffle(buffer_size=BUFFER_SIZE).batch(BATCH_SIZE)
    
    ae_path = os.path.join("tensorflow", "MNIST_autoencoder")
    if not os.path.exists(ae_path):
        os.makedirs(ae_path)

    # Define latent dimension.
    LATENT_DIM = 64
    EPOCHS = 50

    # Define autoencoder.
    ae = AE(encoder=MNISTEncoder(latent_dim=LATENT_DIM),
            decoder=MNISTDecoder(out_dim=x_train_feature_mask.shape[1]))

    # Define optimizer and loss function.

    optimizer = keras.optimizers.Adam(learning_rate=1e-3)
    loss = keras.losses.BinaryCrossentropy(from_logits=False)

    # Compile autoencoder.
    ae.compile(optimizer=optimizer, loss=loss)

    # if len(os.listdir(ae_path)) == 0:
        # Fit and save autoencoder.
    if len(os.listdir(ae_path)) == 0: 
        ae.fit(trainset_ae, epochs=EPOCHS)
        ae.save_weights(ae_path+'/mnist_test_save.h5')
    else:
        ae.build(input_shape = x_train_feature_mask.shape)
        ae.load_weights(ae_path+'/mnist_test_save.h5')
    # else:
    #     # Load the model.
    # ae = keras.models.load_model(ae_path)

    # Define constants
    COEFF_SPARSITY = 7.5                 # sparisty coefficient
    COEFF_CONSISTENCY = 0                # consisteny coefficient -> no consistency
    TRAIN_STEPS = 50000                  # number of training steps -> consider increasing the number of steps
    BATCH_SIZE = 100                    # batch size    

    # shape = (1,) + x_train_feature_mask.shape[1:]

    explainer = CounterfactualRL(predictor=model_test,
                             encoder=ae.encoder,
                             decoder=ae.decoder,
                             latent_dim=LATENT_DIM,
                             coeff_sparsity=COEFF_SPARSITY,
                             coeff_consistency=COEFF_CONSISTENCY,
                             train_steps=TRAIN_STEPS,
                             batch_size=BATCH_SIZE,
                             backend="tensorflow")
    explainer.fit(x_train_feature_mask)
    
    explanation = explainer.explain(
        cf_from[:, mask], Y_t=np.array([0]), batch_size=100)
 
    model_test.forward = model_test.forward_save

    try:
        exp_base = np.zeros(shape=(explanation.cf['X'].shape[0], x_train.shape[1]))
        exp_base[:,mask] = explanation.cf['X']
        return exp_base
    except:
        exp_base = np.zeros(shape=(1, x_train.shape[1]))
        return exp_base


def CF_case_study_abili_Proto(
    model,
    data_use,
    datashow_lat,
    cf_index_to=0,
    cf_index_from=[1,2,4,5,6,7,8,12,16,19],
    n_cf=1,
    num_data=5000,
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
        print('exp {}'.format(i))
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


def CF_case_study_abili_Proto_ae(
    model,
    cf_index_to=[0],
    cf_index_from=[1,2,4,5,6,7,8,12,16,19],
    n_cf=1,
    num_data=5000,
    beta=0.1,
    kappa=0.0,
    theta=100.0,
):

    data_use = gpu2np(model.data_train.data[:num_data])
    img_from = data_use[cf_index_from]
    img_to = data_use[cf_index_to]

    datashow_mid = gpu2np(model(model.data_train.data[:num_data])[2])
    # dict_cf = {}
    cf_data = GenCF_case_stydy_alibi_Proto_ae(
        model=model, 
        cf_from=gpu2np(model.data_train.data[cf_index_from]),
        cf_aim=datashow_mid[cf_index_to],
        beta=beta,
        kappa=kappa,
        theta=theta,
        # sample_size=10000,
        )

    return img_to, img_from, cf_data



def GenCF_case_stydy_alibi_Proto_ae(
    model, 
    cf_from, 
    cf_aim=[0,0], 
    num_data=10000,
    beta=0.1,
    kappa=0.0,
    theta=100.,
    max_iterations=500,
    ):
    tf.compat.v1.disable_v2_behavior()
    # model.eval()
    model_test = model
    
    x_train = model_test.data_train.data
    x_train = gpu2np(x_train.reshape((model_test.data_train.data.shape[0],-1)))
    model_test.forward_save = model_test.forward
    model_test.forward = model_test.predict_

    # X = gpu2np(model_test.data_train.data[cf_index_from : cf_index_from + 1])
    # model_test.cf_aim = cf_aim

    # predict_fn = lambda x: model_test.predict_

    mask = gpu2np(model_test.mask>0)
    x_train_feature_mask = x_train[:, mask]
    # X_after_mask = X[:, mask]
    
    # ------
    BATCH_SIZE = 64
    BUFFER_SIZE = 1024
    trainset_ae = tf.data.Dataset.from_tensor_slices(x_train_feature_mask)
    trainset_ae = trainset_ae.map(lambda x: (x, x))
    trainset_ae = trainset_ae.shuffle(buffer_size=BUFFER_SIZE).batch(BATCH_SIZE)

    # Define autoencode testset.
    testset_ae = tf.data.Dataset.from_tensor_slices(x_train_feature_mask)
    testset_ae = testset_ae.map(lambda x: (x, x))
    testset_ae = testset_ae.shuffle(buffer_size=BUFFER_SIZE).batch(BATCH_SIZE)
    
    ae_path = os.path.join("tensorflow", "MNIST_autoencoder")
    if not os.path.exists(ae_path):
        os.makedirs(ae_path)

    # Define latent dimension.
    LATENT_DIM = 64
    EPOCHS = 50

    # Define autoencoder.
    # ae = AE(encoder=MNISTEncoder(latent_dim=LATENT_DIM),
    #         decoder=MNISTDecoder(out_dim=x_train_feature_mask.shape[1]))

    # # Define optimizer and loss function.

    # optimizer = keras.optimizers.Adam(learning_rate=1e-3)
    # loss = keras.losses.BinaryCrossentropy(from_logits=False)

    # # Compile autoencoder.
    # ae.compile(optimizer=optimizer, loss=loss)


    def ae_model(input_shape, latent_shape):
        # encoder
        x_in = Input(shape=(input_shape))
        x = Dense(1000, activation='relu')(x_in)
        x = Dense(1000, activation='relu')(x)
        encoded = Dense(latent_shape, activation='relu')(x)
        encoder = Model(x_in, encoded)

        # decoder
        x_lat = Input(shape=(latent_shape))
        x = Dense(1000, activation='relu')(x_lat)
        x = Dense(1000, activation='relu')(x)
        decoded = Dense(input_shape, activation='relu')(x)
        decoder = Model(x_lat, decoded)

        # autoencoder = encoder + decoder
        x_out = decoder(encoder(x_in))
        autoencoder = Model(x_in, x_out)
        autoencoder.compile(optimizer='adam', loss='mse')

        return autoencoder, encoder, decoder

    ae, enc, dec = ae_model(
        input_shape=x_train_feature_mask.shape[1], 
        latent_shape=50)
    ae.fit(
        x_train_feature_mask, 
        x_train_feature_mask, 
        batch_size=128, 
        epochs=4, 
        validation_data=(x_train_feature_mask, x_train_feature_mask), 
        verbose=0
        )

    # # if len(os.listdir(ae_path)) == 0:
    #     # Fit and save autoencoder.
    # if len(os.listdir(ae_path)) == 0: 
    #     ae.fit(trainset_ae, epochs=EPOCHS)
    #     ae.save_weights(ae_path+'/mnist_test_save.h5')
    # else:
    #     ae.build(input_shape = x_train_feature_mask.shape)
    #     ae.load_weights(ae_path+'/mnist_test_save.h5')
    # else:
    #     # Load the model.
    # ae = keras.models.load_model(ae_path)

    # Define constants
    COEFF_SPARSITY = 7.5                 # sparisty coefficient
    COEFF_CONSISTENCY = 0                # consisteny coefficient -> no consistency
    TRAIN_STEPS = 50000                  # number of training steps -> consider increasing the number of steps
    BATCH_SIZE = 100                     # batch size    

    shape = (1,) + x_train_feature_mask.shape[1:]

    # explainer = CounterfactualProto(
    #     model_test, shape,
    #     use_kdtree=True, 
    #     # kappa=kappa,
    #     gamma=100.,
    #     theta=theta,
    #     c_init=1.,
    #     c_steps=2,
    #     beta=beta,
    #     max_iterations=max_iterations,
    #     feature_range=(x_train_feature_mask.min(), x_train_feature_mask.max())
    #     )
    explainer = CounterfactualProto(
        model_test, shape, 
        gamma=100., theta=theta,
        ae_model=ae, enc_model=enc, 
        max_iterations=max_iterations,
        feature_range=(x_train_feature_mask.min(), x_train_feature_mask.max()),
        c_init=1.,
        c_steps=2,
        )

    explainer.fit(x_train_feature_mask[:num_data])
    
    exp_r = []
    for i in range(cf_from.shape[0]):
        print('exp {}'.format(i))
        explanation = explainer.explain(
            cf_from[i:i+1, mask],
            )
        try:
            exp_r.append(explanation.cf['X'])
        except:
            exp_r.append(np.zeros_like(cf_from[i:i+1, mask]))

    # explainer.fit(x_train_feature_mask)
    exp_r = np.concatenate(exp_r)
    # explanation = explainer.explain(
    #     cf_from[:, mask], Y_t=np.array([0]), batch_size=100)
 
    model_test.forward = model_test.forward_save

    try:
        exp_base = np.zeros(shape=(exp_r.shape[0], x_train.shape[1]))
        exp_base[:,mask] = exp_r
        return exp_base
    except:
        exp_base = np.zeros(shape=(1, x_train.shape[1]))
        return exp_base


