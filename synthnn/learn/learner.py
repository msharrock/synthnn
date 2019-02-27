#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
synthnn.learn.learner

train functions for synthnn neural networks

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)

Created on: Feb 25, 2018
"""

__all__ = ['get_data_augmentation',
           'get_dataloader',
           'get_device',
           'get_model',
           'Learner']

from typing import List, Optional, Union

import logging
import os

import nibabel as nib
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
from torch.utils.data.sampler import SubsetRandomSampler

from niftidataset import MultimodalNiftiDataset, MultimodalImageDataset, split_filename
import niftidataset.transforms as niftitfms

from ..errors import SynthNNError
from .optim import BurnCosineLR
from ..plot.loss import plot_loss
from .predict import Predictor
from ..util.config import ExperimentConfig
from ..util.helper import get_optim, init_weights

logger = logging.getLogger(__name__)


class Learner:

    def __init__(self, model, device=None, train_loader=None, valid_loader=None, optimizer=None, predictor=None):
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.optimizer = optimizer
        self.device = device
        self.predictor = predictor

    @classmethod
    def train_setup(cls, config:Union[str,ExperimentConfig]):
        if isinstance(config,str):
            config = ExperimentConfig.load_json(config)
        device, use_cuda = get_device(config.gpu_selector, config.disable_cuda)
        model = get_model(config, True, device)
        logger.debug(model)
        logger.info(f'Number of trainable parameters in model: {get_num_params(model)}')
        if os.path.isfile(config.trained_model):
            model, start_epoch = load_model(model, config.trained_model, device)
            logger.info(f'Loaded checkpoint: {config.trained_model} (epoch {start_epoch})')
        else:
            logger.info(f'Initializing weights with {config.init}')
            init_weights(model, config.init, config.init_gain)
        if use_cuda: model.cuda(device=device)
        train_loader, valid_loader = get_dataloader(config)
        logger.info(('Max ' if config.lr_scheduler else '') + f'LR: {config.learning_rate:.5f}')
        optimizer = get_optim(config.optimizer)(model.parameters(), lr=config.learning_rate,
                                                weight_decay=config.weight_decay)
        if os.path.isfile(config.trained_model) and not config.no_load_opt:
            optimizer = load_opt(optimizer, config.trained_model)
        model.train()
        predictor = Predictor(model, config.patch_size, config.batch_size, device, config.sample_axis,
                              config.n_output, config.net3d)
        return cls(model, device, train_loader, valid_loader, optimizer, predictor)

    @classmethod
    def predict_setup(cls, config:Union[str,ExperimentConfig]):
        if isinstance(config,str):
            config = ExperimentConfig.load_json(config)
        device, use_cuda = get_device(config.gpu_selector, config.disable_cuda)
        nsyn = config.monte_carlo or 1
        model = get_model(config, nsyn > 1 and config.dropout_prob > 0, device)
        logger.debug(model)
        model, _ = load_model(model, config.trained_model, device)
        if use_cuda: model.cuda(device=device)
        model.eval()
        predictor = Predictor(model, config.patch_size, config.batch_size, device, config.sample_axis,
                              config.n_output, config.net3d)
        return cls(model, device, predictor=predictor)

    def fit(self, n_epochs, clip:float=None, checkpoint:int=None, trained_model:str=None):
        """ training loop for neural network """
        self.model.train()
        use_valid = self.valid_loader is not None
        train_losses, valid_losses = [], []
        for t in range(1, n_epochs + 1):
            # training
            t_losses = []
            if use_valid: self.model.train(True)
            for src, tgt in self.train_loader:
                src, tgt = src.to(self.device), tgt.to(self.device)
                self.optimizer.zero_grad()
                out = self.model(src)
                loss = self._criterion(out, tgt)
                t_losses.append(loss.item())
                if hasattr(self, 'amp_handle'):
                    with self.amp_handle.scale_loss(loss, self.optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()
                if clip is not None: nn.utils.clip_grad_norm_(self.model.parameters(), clip)
                self.optimizer.step()
            train_losses.append(t_losses)
            if hasattr(self, 'scheduler'): self.scheduler.step()

            if checkpoint is not None:
                if t % checkpoint == 0:
                    path, base, ext = split_filename(trained_model)
                    fn = os.path.join(path, base + f'_chk_{t}' + ext)
                    self.save(fn, t)

            # validation
            v_losses = []
            if use_valid: self.model.train(False)
            with torch.set_grad_enabled(False):
                for src, tgt in self.valid_loader:
                    src, tgt = src.to(self.device), tgt.to(self.device)
                    out = self.model(src)
                    loss = self._criterion(out, tgt)
                    v_losses.append(loss.item())
                valid_losses.append(v_losses)

            if np.any(np.isnan(t_losses)): raise SynthNNError('NaN in training loss, cannot recover. Exiting.')
            if logger is not None:
                log = f'Epoch: {t} - Training Loss: {np.mean(t_losses):.2e}'
                if use_valid: log += f', Validation Loss: {np.mean(v_losses):.2e}'
                if hasattr(self, 'scheduler'): log += f', LR: {self.scheduler.get_lr()[0]:.2e}'
                logger.info(log)
        return train_losses, valid_losses

    def predict(self, fn:str, nsyn:int=1, temperature_map:bool=False, calc_var:bool=False):
        self.model.eval()
        f = fn[0].lower()
        if f.endswith('.nii') or f.endswith('.nii.gz'):
            img_nib = nib.load(fn[0])
            img = np.stack([np.asarray(nib.load(f).get_data(), dtype=np.float32) for f in fn])
            out = self.predictor.predict(img, nsyn, temperature_map, calc_var)
            out_img = [nib.Nifti1Image(out[i], img_nib.affine, img_nib.header) for i, _ in enumerate(fn)]
        elif f.endswith('.tif') or f.endswith('.tiff'):
            out = self.predictor.img_predict(fn, nsyn, temperature_map, calc_var)
            out_img = Image.fromarray(out)
        elif f.endswith('.png'):
            out = self.predictor.png_predict(fn, nsyn, temperature_map, calc_var)
            out_img = Image.fromarray(out)
        else:
            raise SynthNNError(f'File: {fn[0]}, not supported.')
        return out_img

    def _criterion(self, out, tgt):
        """ helper function to handle multiple outputs in model evaluation """
        c = self.model.module.criterion if isinstance(self.model,nn.DataParallel) else self.model.criterion
        return c(out, tgt)

    def fp16(self):
        """ import and initialize mixed precision training package """
        try:
            from apex import amp
            self.amp_handle = amp.init()
        except ImportError:
            logger.info('Mixed precision training (i.e., the package `apex`) not available.')

    def multigpu(self, gpu_selector:List=None):
        """ put the model on the GPU if available and desired """
        n_gpus = torch.cuda.device_count()
        if n_gpus <= 1:
            logger.warning('Multi-GPU functionality is not available on your system.')
        else:
            n_gpus = len(gpu_selector) if gpu_selector is not None else n_gpus
            logger.info(f'Enabling use of {n_gpus} gpus')
            self.model = torch.nn.DataParallel(self.model, device_ids=gpu_selector)

    def lr_scheduler(self, n_epochs):
        logger.info('Enabling burn-in cosine annealing LR scheduler')
        self.scheduler = BurnCosineLR(self.optimizer, n_epochs)


    def load(self, fn):
        self.model, _ = load_model(self.model, fn, self.device)
        self.optimizer = load_opt(self.optimizer, fn)

    def save(self, fn, epoch=0):
        """ save a model, an optimizer state and the epoch number to a file """
        state_dict = self.model.module.state_dict() if hasattr(self.model, 'module') else self.model.state_dict()
        state = {'epoch': epoch, 'state_dict': state_dict, 'optimizer': self.optimizer.state_dict()}
        torch.save(state, fn)

    @staticmethod
    def plot_loss(train_loss:List, valid_loss:List, fn:str=None, plot_error:bool=False):
        """ plot training and validation losses on the same plot (with or without error bars) """
        ax = plot_loss(train_loss, ecolor='darkorchid', label='Train', plot_error=plot_error)
        _ = plot_loss(valid_loss, filename=fn, ecolor='firebrick', ax=ax, label='Validation', plot_error=plot_error)

    @staticmethod
    def write_csv(train_loss:List, valid_loss:List, fn:str):
        """ write training and validation losses to a csv file """
        import csv
        head = ['epochs','avg train','std train','avg valid','std valid']
        epochs = list(range(1, len(train_loss) + 1))
        avg_tl = [np.mean(losses) for losses in train_loss]
        std_tl = [np.std(losses) for losses in train_loss]
        avg_vl = [np.mean(losses) for losses in valid_loss]
        std_vl = [np.std(losses) for losses in valid_loss]
        out = np.vstack([epochs, avg_tl, std_tl, avg_vl, std_vl]).T
        with open(fn, "w") as f:
            wr = csv.writer(f)
            wr.writerow(head)
            wr.writerows(out)


def get_model(config:ExperimentConfig, enable_dropout:bool=True, device:Optional[torch.device]=None):
    """
    instantiate a model based on an ExperimentConfig class instance

    Args:
        config (ExperimentConfig): instance of the ExperimentConfig class
        enable_dropout (bool): enable dropout in the model (usually for training)
        device (torch.device): device to put the torch tensors on

    Returns:
        model: instance of one of the available models in the synthnn package
    """
    if config.nn_arch == 'nconv':
        from ..models.nconvnet import SimpleConvNet
        logger.warning('The nconv network is for basic testing.')
        model = SimpleConvNet(config.n_layers, kernel_size=config.kernel_size, dropout_p=config.dropout_prob,
                              n_input=config.n_input, n_output=config.n_output, is_3d=config.net3d)
    elif config.nn_arch == 'unet':
        from ..models.unet import Unet
        model = Unet(config.n_layers, kernel_size=config.kernel_size, dropout_p=config.dropout_prob,
                     channel_base_power=config.channel_base_power, add_two_up=config.add_two_up, normalization=config.normalization,
                     activation=config.activation, output_activation=config.out_activation, interp_mode=config.interp_mode,
                     enable_dropout=enable_dropout, enable_bias=config.enable_bias, is_3d=config.net3d,
                     n_input=config.n_input, n_output=config.n_output, no_skip=config.no_skip,
                     ord_params=config.ord_params, noise_lvl=config.noise_lvl, device=device,
                     loss=config.loss, attention=config.attention)
    elif config.nn_arch == 'vae':
        from ..models.vae import VAE
        model = VAE(config.n_layers, config.img_dim, channel_base_power=config.channel_base_power, activation=config.activation,
                    is_3d=config.net3d, n_input=config.n_input, n_output=config.n_output, latent_size=config.latent_size)
    else:
        raise SynthNNError(f'Invalid NN type: {config.nn_arch}. {{nconv, unet, vae}} are the only supported options.')
    return model


def get_device(gpu_selector=None, disable_cuda=False):
    """ get the device(s) for tensors to be put on """
    cuda_avail = torch.cuda.is_available()
    use_cuda = cuda_avail and not disable_cuda
    if use_cuda: torch.backends.cudnn.benchmark = True
    if not cuda_avail and not disable_cuda: logger.warning('CUDA does not appear to be available on your system.')
    n_gpus = torch.cuda.device_count()
    if gpu_selector is not None:
        if len(gpu_selector) > n_gpus or any([gpu_id >= n_gpus for gpu_id in gpu_selector]):
            raise SynthNNError('Invalid number of gpus or invalid GPU ID input in --gpu-selector')
        cuda = f"cuda:{gpu_selector[0]}"  # arbitrarily choose first GPU given
    else:
        cuda = "cuda"
    device = torch.device(cuda if use_cuda else "cpu")
    return device, use_cuda


def get_dataloader(config:ExperimentConfig, tfms:List=None):
    """ get the dataloaders for training/validation """
    # get data augmentation if not defined
    if tfms is None: tfms = get_data_augmentation(config)

    # check number of jobs requested and CPUs available
    num_cpus = os.cpu_count()
    if num_cpus < config.n_jobs:
        logger.warning(f'Requested more workers than available (n_jobs={config.n_jobs}, # cpus={num_cpus}). '
                       f'Setting n_jobs={num_cpus}.')
        config.n_jobs = num_cpus

    # define dataset and split into training/validation set
    use_nii_ds = config.ext is None or 'nii' in config.ext
    dataset = MultimodalNiftiDataset(config.source_dir, config.target_dir, Compose(tfms)) if use_nii_ds else \
              MultimodalImageDataset(config.source_dir, config.target_dir, Compose(tfms), ext='*.' + config.ext)
    logger.info(f'Number of training images: {len(dataset)}')

    if config.valid_source_dir is not None and config.valid_target_dir is not None:
        valid_dataset = MultimodalNiftiDataset(config.valid_source_dir, config.valid_target_dir,
                                               Compose(tfms)) if use_nii_ds else \
                        MultimodalImageDataset(config.valid_source_dir, config.valid_target_dir,
                                               Compose(tfms), ext='*.' + config.ext)
        logger.info(f'Number of validation images: {len(valid_dataset)}')
        train_loader = DataLoader(dataset, batch_size=config.batch_size, num_workers=config.n_jobs, shuffle=True,
                                  pin_memory=config.pin_memory)
        valid_loader = DataLoader(valid_dataset, batch_size=config.batch_size, num_workers=config.n_jobs,
                                  pin_memory=config.pin_memory)
    else:
        # setup training and validation set
        num_train = len(dataset)
        indices = list(range(num_train))
        split = int(config.valid_split * num_train)
        valid_idx = np.random.choice(indices, size=split, replace=False)
        train_idx = list(set(indices) - set(valid_idx))
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)
        # set up data loader for nifti images
        train_loader = DataLoader(dataset, sampler=train_sampler, batch_size=config.batch_size,
                                  num_workers=config.n_jobs, pin_memory=config.pin_memory)
        valid_loader = DataLoader(dataset, sampler=valid_sampler, batch_size=config.batch_size,
                                  num_workers=config.n_jobs, pin_memory=config.pin_memory)

    return train_loader, valid_loader


def get_data_augmentation(config:ExperimentConfig):
    """ get all data augmentation transforms for training """
    # control random cropping patch size (or if used at all)
    if config.ext is None:
        cropper = niftitfms.RandomCrop3D(config.patch_size) if config.net3d else \
                  niftitfms.RandomCrop2D(config.patch_size, config.sample_axis)
        tfms = [cropper] if config.patch_size > 0 else \
               [] if config.net3d else \
               [niftitfms.RandomSlice(config.sample_axis)]
    else:
        tfms = [niftitfms.RandomCrop(config.patch_size)] if config.patch_size > 0 else []

    # add data augmentation if desired
    if config.prob is not None:
        logger.info('Adding data augmentation transforms')
        tfms.extend(niftitfms.get_transforms(config.prob, config.tfm_x, config.tfm_y, config.rotate, config.translate,
                                             config.scale, config.vflip, config.hflip, config.gamma, config.gain,
                                             config.noise_pwr, config.block, config.mean, config.std))
    else:
        logger.info('No data augmentation will be used (except random cropping if patch_size > 0)')
        tfms.append(niftitfms.ToTensor())

    return tfms


def load_model(model, fn, device):
    """
    load a model's weights

    Args:
        model (torch.nn.Module): instance of a synthnn model
        fn (str): filename associated with model weights
        device (torch.device): device to put the model on

    Returns:
        model, last_epoch: fills in the weights and returns the last epoch
    """
    checkpoint = torch.load(fn)
    last_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device)
    return model, last_epoch


def load_opt(optimizer, fn):
    """ load an optimizer state """
    checkpoint = torch.load(fn)
    optimizer.load_state_dict(checkpoint['optimizer'])
    return optimizer


def get_num_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)