#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
synthnn.models.unet

holds the architecture for a 2d or 3d unet [1]

References:
    [1] O. Cicek, A. Abdulkadir, S. S. Lienkamp, T. Brox, and O. Ronneberger,
        “3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation,”
        in Medical Image Computing and Computer-Assisted Intervention (MICCAI), 2016, pp. 424–432.

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)

Created on: Nov 2, 2018
"""

__all__ = ['Unet']

import logging
from typing import Optional, Tuple, Union

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from synthnn import get_act, get_norm3d, get_norm2d, SynthNNError

logger = logging.getLogger(__name__)


class Unet(torch.nn.Module):
    """
    defines a 2d or 3d unet [1] in pytorch

    Args:
        n_layers (int): number of layers (to go down and up)
        kernel_size (int): size of kernel (symmetric)
        dropout_p (int): dropout probability for each layer
        channel_base_power (int): 2 ** channel_base_power is the number of channels in the first layer
            and increases in each proceeding layer such that in the n-th layer there are
            2 ** channel_base_power + n channels (this follows the convention in [1])
        add_two_up (bool): flag to add two to the kernel size on the upsampling following
            the paper [2]
        normalization_layer (str): type of normalization layer to use (batch or [instance])
        activation (str): type of activation to use throughout network except final ([relu], lrelu, linear, sigmoid, tanh)
        output_activation (str): final activation in network (relu, lrelu, [linear], sigmoid, tanh)
        is_3d (bool): if false define a 2d unet, otherwise the network is 3d
        interp_mode (str): use one of {'nearest', 'bilinear', 'trilinear'} for upsampling interpolation method
            depending on if the unet is 3d or 2d
        enable_dropout (bool): enable the use of dropout (if dropout_p is set to zero then there will be no dropout,
            however if this is false and dropout_p > 0, then no dropout will be used) [Default=True]
        enable_bias (bool): enable bias calculation in final and upsampconv layers [Default=False]
        n_input (int): number of input channels to network [Default=1]
        n_output (int): number of output channels for network [Default=1]
        no_skip (bool): use no skip connections [Default=False]
        ord_params (Tuple[int,int,int]): parameters for ordinal regression (start,end,n_bins) [Default=None]
        noise_lvl (float): add gaussian noise to weights with this std [Default=0]
        device (torch.device): device to place new parameters/tensors on (only necessary for ord_params or noise_lvl)
            [Default=None]

    References:
        [1] O. Cicek, A. Abdulkadir, S. S. Lienkamp, T. Brox, and O. Ronneberger,
            “3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation,”
            in Medical Image Computing and Computer-Assisted Intervention (MICCAI), 2016, pp. 424–432.
        [2] C. Zhao, A. Carass, J. Lee, Y. He, and J. L. Prince, “Whole Brain Segmentation and Labeling
            from CT Using Synthetic MR Images,” MLMI, vol. 10541, pp. 291–298, 2017.

    """
    def __init__(self, n_layers:int, kernel_size:int=3, dropout_p:float=0, channel_base_power:int=5,
                 add_two_up:bool=False, normalization:str='instance', activation:str='relu', output_activation:str='linear',
                 is_3d:bool=True, interp_mode:str='nearest', enable_dropout:bool=True,
                 enable_bias:bool=False, n_input:int=1, n_output:int=1, no_skip:bool=False,
                 ord_params:Tuple[int,int,int]=None, noise_lvl:float=0, device:torch.device=None):
        super(Unet, self).__init__()
        # setup and store instance parameters
        self.n_layers = n_layers
        self.kernel_sz = kernel_size
        self.dropout_p = dropout_p
        self.channel_base_power = channel_base_power
        self.a2u = 2 if add_two_up else 0
        self.norm = nm = normalization
        self.act = a = activation
        self.out_act = oa = output_activation
        self.is_3d = is_3d
        self.interp_mode = interp_mode
        self.enable_dropout = enable_dropout
        self.enable_bias = enable_bias
        self.n_input = n_input
        self.n_output = n_output
        self.no_skip = no_skip
        self.ord_params = ord_params
        self.noise_lvl = noise_lvl
        self.device = device
        self.criterion = nn.MSELoss() if ord_params is None else _OrdLoss(ord_params, device, is_3d)
        nl = n_layers - 1
        def lc(n): return int(2 ** (channel_base_power + n))  # shortcut to layer channel count
        # define the model layers here to make them visible for autograd
        self.start = self._unet_blk(n_input, lc(0), lc(0), act=(a, a), norm=(nm, nm))
        self.down_layers = nn.ModuleList([self._unet_blk(lc(n), lc(n+1), lc(n+1), act=(a, a), norm=(nm, nm))
                                          for n in range(nl)])
        self.bridge = self._unet_blk(lc(nl), lc(nl+1), lc(nl+1), act=(a, a), norm=(nm, nm))
        self.up_layers = nn.ModuleList([self._unet_blk(lc(n) + lc(n) if not no_skip else lc(n),
                                                       lc(n), lc(n), (kernel_size+self.a2u, kernel_size),
                                                       act=(a, a), norm=(nm, nm))
                                        for n in reversed(range(1,nl+1))])
        self.finish = self._final(lc(0) + n_input if not no_skip else lc(0), n_output, oa, bias=enable_bias)
        self.upsampconvs = nn.ModuleList([self._conv(lc(n+1), lc(n), 3, bias=enable_bias)
                                          for n in reversed(range(nl+1))])

    def forward(self, x:torch.Tensor, return_var:bool=False) -> torch.Tensor:
        x = self._fwd_skip(x, return_var) if not self.no_skip else self._fwd_no_skip(x, return_var)
        return x

    def _fwd_skip(self, x:torch.Tensor, return_var:bool=False) -> torch.Tensor:
        dout = [x]
        dout.append(self.start(x))
        x = self._down(dout[-1])
        for dl in self.down_layers:
            dout.append(dl(x))
            x = self._add_noise(self._down(dout[-1]))
        x = self.upsampconvs[0](self._add_noise(self._up(self.bridge(x), dout[-1].shape[2:])))
        for i, (ul, d) in enumerate(zip(self.up_layers, reversed(dout)), 1):
            x = ul(torch.cat((x, d), dim=1))
            x = self._add_noise(self._up(x, dout[-i-1].shape[2:]))
            x = self.upsampconvs[i](x)
        if not return_var:
            x = self.finish(torch.cat((x, dout[0]), dim=1)) if not isinstance(self.finish,nn.ModuleList) else \
                self.finish[0](torch.cat((x, dout[0]), dim=1)) / self.finish[1](x)
        else:
            x = self.finish[1](x)
        return x

    def _fwd_no_skip(self, x:torch.Tensor, return_var:bool=False) -> torch.Tensor:
        sz = [x.shape]
        x = self.start(x)
        x = self._down(x)
        for dl in self.down_layers:
            x = dl(x)
            sz.append(x.shape)
            x = self._add_noise(self._down(x))
        x = self.upsampconvs[0](self._add_noise(self._up(self.bridge(x), sz[-1][2:])))
        for i, (ul, s) in enumerate(zip(self.up_layers, reversed(sz)), 1):
            x = ul(x)
            x = self._add_noise(self._up(x, sz[-i-1][2:]))
            x = self.upsampconvs[i](x)
        if not return_var:
            x = self.finish(x) if not isinstance(self.finish,nn.ModuleList) else self.finish[0](x) / self.finish[1](x)
        else:
            x = self.finish[1](x)
        return x

    def _down(self, x:torch.Tensor) -> torch.Tensor:
        y = (F.max_pool3d(x, (2,2,2)) if self.is_3d else F.max_pool2d(x, (2,2)))
        return y

    def _up(self, x:torch.Tensor, sz:Union[Tuple[int,int,int], Tuple[int,int]]) -> torch.Tensor:
        y = F.interpolate(x, size=sz, mode=self.interp_mode)
        return y

    def _add_noise(self, x:torch.Tensor) -> torch.Tensor:
        if self.dropout_p > 0:
            x = F.dropout3d(x, self.dropout_p, training=self.enable_dropout) if self.is_3d else \
                F.dropout2d(x, self.dropout_p, training=self.enable_dropout)
        if self.noise_lvl > 0:
            x.add_(torch.randn(x.size()).to(self.device) * self.noise_lvl)
        return x

    def _conv(self, in_c:int, out_c:int, kernel_sz:Optional[int]=None, mode:str=None, bias:bool=False) -> nn.Sequential:
        ksz = self.kernel_sz if kernel_sz is None else kernel_sz
        stride = 1 if mode is None else 2
        bias = False if self.norm != 'none' and not bias else True
        c = nn.Sequential(nn.ReplicationPad3d(ksz // 2), nn.Conv3d(in_c, out_c, ksz, stride=stride, bias=bias)) if self.is_3d else \
            nn.Sequential(nn.ReflectionPad2d(ksz // 2),  nn.Conv2d(in_c, out_c, ksz, stride=stride, bias=bias))
        return c

    def _conv_act(self, in_c:int, out_c:int, kernel_sz:Optional[int]=None,
                  act:Optional[str]=None, norm:Optional[str]=None, mode:str=None) -> nn.Sequential:
        ksz = self.kernel_sz if kernel_sz is None else kernel_sz
        activation = get_act(act) if act is not None else get_act('relu')
        normalization = get_norm3d(norm, out_c) if norm is not None and self.is_3d else get_norm3d('instance', out_c) if self.is_3d else \
                        get_norm2d(norm, out_c) if norm is not None and not self.is_3d else get_norm2d('instance', out_c)
        layers = [self._conv(in_c, out_c, ksz, mode)]
        if normalization is not None: layers.append(normalization)
        layers.append(activation)
        ca = nn.Sequential(*layers)
        return ca

    def _unet_blk(self, in_c:int, mid_c:int, out_c:int,
                  kernel_sz:Tuple[Optional[int],Optional[int]]=(None,None),
                  act:Tuple[Optional[str],Optional[str]]=(None,None),
                  norm:Tuple[Optional[str],Optional[str]]=(None,None)) -> nn.Sequential:
        dca = nn.Sequential(
            self._conv_act(in_c,  mid_c, kernel_sz[0], act[0], norm[0]),
            self._conv_act(mid_c, out_c, kernel_sz[1], act[1], norm[1]))
        return dca

    def _final(self, in_c:int, out_c:int, out_act:Optional[str]=None, bias:bool=False):
        if self.ord_params is None:
            c = self._conv(in_c, out_c, 1, bias=bias)
            fc = nn.Sequential(c, get_act(out_act)) if out_act != 'linear' else c
            return fc
        else:
            n_classes = self.ord_params[2]
            fc = self._conv(in_c, n_classes, 1, bias=bias)
            fc_temp_c = self._conv(in_c, out_c, 1, bias=bias) if self.no_skip else \
                        self._conv(in_c - self.n_input, 1, bias=bias)  # temp should not be a residual layer
            fc_temp = nn.Sequential(fc_temp_c, nn.Softplus())
            return nn.ModuleList([fc, fc_temp])

    def predict(self, x:torch.Tensor, return_var:bool=False) -> torch.Tensor:
        if self.ord_params is None:
            return self.forward(x)
        else:
            y_hat = self.forward(x, return_var)
            if not return_var: y_hat = self.criterion.predict(y_hat)
            return y_hat


class _OrdLoss(nn.Module):
    def __init__(self, params:Tuple[int,int,int], device:torch.device, is_3d:bool=False):
        super(_OrdLoss, self).__init__()
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
