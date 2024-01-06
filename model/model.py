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