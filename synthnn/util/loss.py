#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
synthnn.util.loss

define general loss functions for neural network training

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)

Created on: Feb 20, 2018
"""

__all__ = ['get_loss',
           'NCCLoss',
           'OrdLoss',
           'VAELoss']

from typing import Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def get_loss(name):
    if name == 'mse' or name is None:
        loss = nn.MSELoss()
    elif name == 'ncc':
        loss = NCCLoss()
    elif name == 'mae':
        loss = nn.L1Loss()
    else:
        raise ValueError(f'Loss function {name} not supported.')
    return loss


class NCCLoss(nn.Module):
    """ normalized cross correlation loss """
    @staticmethod
    def _standardize(x):
        xm = x.mean()
        return (x - xm) / torch.norm(x - xm)

    def forward(self, y:torch.Tensor, y_hat:torch.Tensor):
        ncc = (self._standardize(y_hat) * self._standardize(y)).sum()
        return 1 - ncc


class OrdLoss(nn.Module):
    def __init__(self, params:Tuple[int,int,int], device:torch.device, is_3d:bool=False):
        super(OrdLoss, self).__init__()
        start, stop, n_bins = params
        self.device = device
        self.bins = np.linspace(start, stop, n_bins-1, endpoint=False)
        self.tbins = self._linspace(start, stop, n_bins, is_3d).to(self.device)
        self.mae = nn.L1Loss()
        self.ce = nn.CrossEntropyLoss()

    @staticmethod
    def _linspace(start:int, stop:int, n_bins:int, is_3d:bool) -> torch.Tensor:
        rng = np.linspace(start, stop, n_bins, dtype=np.float32)
        trng = torch.from_numpy(rng[:,None,None])
        return trng if not is_3d else trng[...,None]

    def _digitize(self, x:torch.Tensor) -> torch.Tensor:
        return torch.from_numpy(np.digitize(x.cpu().detach().numpy(), self.bins)).squeeze().to(self.device)

    def predict(self, yd_hat:torch.Tensor) -> torch.Tensor:
        p = F.softmax(yd_hat, dim=1)
        intensity_bins = torch.ones_like(yd_hat) * self.tbins
        y_hat = torch.sum(p * intensity_bins, dim=1, keepdim=True)
        return y_hat

    def forward(self, y:torch.Tensor, yd_hat:torch.Tensor):
        yd = self._digitize(y)
        CE = self.ce(yd_hat, yd)
        y_hat = self.predict(yd_hat)
        MAE = self.mae(y_hat, y)
        return CE + MAE


class VAELoss(nn.Module):
    def __init__(self):
        super(VAELoss, self).__init__()
        self.mse_loss = nn.MSELoss(reduction="sum")

    def forward(self, x, recon_x, mu, logvar):
        MSE = self.mse_loss(recon_x, x)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2)-logvar.exp())
        return MSE + KLD
