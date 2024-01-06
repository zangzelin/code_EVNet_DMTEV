from curses import color_content
import os
import sys
import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn

import wandb
from aug.aug import aug_near_feautee_change, aug_near_mix, aug_randn
from dataloader import data_base
import patemb_main
import plotly.graph_objects as go
import plotly.express as px
import eval.eval_core as ec

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


def main(args):

    pl.utilities.seed.seed_everything(1)
    info = [str(s) for s in sys.argv[1:]]
    runname = "_".join(["dmt", args.data_name, "".join(info)])
    # checkpoint_callback = []

    wandb.init(
        name=runname,
        project="PatEmb" + args.project_name,
        entity="zangzelin",
        mode="offline" if bool(args.offline) else "online",
        save_code=True,
        config=args,
    )

    model = patemb_main.LitPatNN(
        dataname=args.data_name,
        **args.__dict__,
    )
    # early_stop = EarlyStopping(
    #     monitor="es_monitor", patience=1, verbose=False, mode="max"
    # )
    # trainer = Trainer(
    #     gpus=1,
    #     max_epochs=args.epochs,
    #     # progress_bar_refresh_rate=0,
    #     progress_bar_refresh_rate=10,
    #     callbacks=[
    #         early_stop,
    #         # checkpoint_callback,
    #         ],
    # )
    print('start fit')

    if args.data_name == 'Digits':
        model = model.load_from_checkpoint(
            checkpoint_path="save_checkpoint_use/Digitsepoch=599.ckpt")
        nppcheckpoint = np.load('save_checkpoint_use/Digits=599.npy')
        model.PM_root.weight.data = torch.tensor(nppcheckpoint)
    if args.data_name == 'Mnist':
        model = model.load_from_checkpoint(
            checkpoint_path="save_checkpoint_use/Mnistepoch=899.ckpt")
        model.PM_root.weight.data = torch.tensor(
            np.load('save_checkpoint_use/Mnist=899.npy')
        )
    model.mask = model.PM_root.weight.reshape(-1) > 0.1
    # import pdb; pdb.set_trace()
    print('end fit')
    model.eval()
    model = model.to('cpu')
    data = torch.tensor(model.data_train.data).to('cpu')
    data = data.reshape(data.shape[0], -1)
    label = torch.tensor(model.data_train.label).to('cpu')
    mask = model.mask.to('cpu')
    ins_emb = gpu2np(model.forward_fea(data)[2])
    index = torch.arange(0, data.shape[0])
    # import numpy as np

    sortx = np.sort(np.copy(ins_emb)[:, 0])
    blockx0, blockx1 = sortx[10], sortx[-10]
    blockx0 = blockx0 - 0.2*(blockx1-blockx0)
    blockx1 = blockx1 + 0.2*(blockx1-blockx0)

    sorty = np.sort(np.copy(ins_emb)[:, 1])
    blocky0, blocky1 = sorty[10], sorty[-10]
    blocky0 = blocky0 - 0.2*(blocky1-blocky0)
    blocky1 = blocky1 + 0.2*(blocky1-blocky0)

    a2 = ins_emb[:, 0] > blockx0
    a1 = ins_emb[:, 0] < blockx1
    a3 = ins_emb[:, 1] < blocky1
    a4 = ins_emb[:, 1] > blocky0

    bool_mask = (
        a1.astype(np.float32) +
        a2.astype(np.float32) +
        a3.astype(np.float32) +
        a4.astype(np.float32)) == 4
    data = data[bool_mask]
    label = label[bool_mask]
    index = index[bool_mask]
    ins_emb = ins_emb[bool_mask]

    if args.showmainfig > 0:
        model.mask = model.mask.to(model.device)
        model.PM_root = model.PM_root.to(model.device)

        wandb.log({'main/fig': model.up_mainfig_case(
            data, ins_emb, label, index, mask=mask, top_cluster=2,
            )})
    else:
        model = model.to('cpu')
        model.mask = model.mask.to(model.device)
        model.PM_root = model.PM_root.to(model.device)
        wandb.log({'main_easy/fig_easy': model.up_mainfig_emb(
            data, ins_emb, label, index, mask=mask)})
    if args.data_name == 'Digits':
        pix = 8
    if args.data_name == 'Mnist':
        pix = 28

    pix_digits = gpu2np(model.PM_root.weight.data)
    m = pix_digits < 0.1
    pix_digits = (
        pix_digits-pix_digits.min()
        )/(pix_digits.max()-pix_digits.min())
    pix_digits[m] = None
    fig = px.imshow(
        pix_digits.reshape((pix, pix)),
        color_continuous_scale='Blues',
        range_color=[0.0, 1.0],
        )
    wandb.log({'zzz': fig})


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
            "PBMCD2638",
            "PBMC",
            "InsEmb_TPD_579_ALL_PRO",
            'InsEmb_TPD_579_ALL_PRO5C',
            'YONGJIE_UC',
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
    parser.add_argument("--l2alpha", type=float, default=10)
    parser.add_argument("--nu", type=float, default=1e-2)
    parser.add_argument("--num_link_aim", type=float, default=0.2)
    # parser.add_argument("--num_fea_aim", type=int, default=128)
    parser.add_argument("--num_fea_aim", type=int, default=50)
    parser.add_argument("--K_plot", type=int, default=40)
    parser.add_argument("--save_checkpoint", type=int, default=0)
    parser.add_argument("--explevel", type=int, default=2)

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
