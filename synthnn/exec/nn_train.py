#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
synthnn.exec.nn_train

command line interface to train a deep convolutional neural network for
synthesis of MR (brain) images

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)

Created on: Aug 28, 2018
"""

import argparse
from collections import OrderedDict
import logging
import os
import sys
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', category=UserWarning)
    import matplotlib
    matplotlib.use('agg')  # do not pull in GUI
    import numpy as np
    import torch
    from torch import nn
    from torch.utils.data import DataLoader
    from torchvision.transforms import Compose
    from torch.utils.data.sampler import SubsetRandomSampler
    from niftidataset import MultimodalNiftiDataset, MultimodalImageDataset
    import niftidataset.transforms as tfms
    from synthnn import SynthNNError, init_weights, BurnCosineLR, split_filename
    from synthnn.util.exec import get_args, get_device, setup_log, write_out_config


######## Helper functions ########

def arg_parser():
    parser = argparse.ArgumentParser(description='train a CNN for MR image synthesis')

    required = parser.add_argument_group('Required')
    required.add_argument('-s', '--source-dir', type=str, required=True, nargs='+',
                          help='path to directory with source images (multiple paths can be provided for multi-modal synthesis)')
    required.add_argument('-t', '--target-dir', type=str, required=True, nargs='+',
                          help='path to directory with target images (multiple paths can be provided for multi-modal synthesis)')
    required.add_argument('-o', '--trained-model', type=str, default=None,
                          help='path to output the trained model or (if model exists) continue training this model')

    options = parser.add_argument_group('Options')
    options.add_argument('-bs', '--batch-size', type=int, default=5,
                         help='batch size (num of images to process at once) [Default=5]')
    options.add_argument('-c', '--clip', type=float, default=None,
                         help='gradient clipping threshold [Default=None]')
    options.add_argument('-chk', '--checkpoint', type=int, default=None,
                         help='save the model every `checkpoint` epochs [Default=None]')
    options.add_argument('--disable-cuda', action='store_true', default=False,
                         help='Disable CUDA regardless of availability')
    options.add_argument('-e', '--ext', type=str, default=None, help='extension for 2d image [Default=None]')
    options.add_argument('-mp', '--fp16', action='store_true', default=False,
                         help='enable mixed precision training')
    options.add_argument('-gs', '--gpu-selector', type=int, nargs='+', default=None,
                         help='use gpu(s) selected here, None uses all available gpus if --multi-gpus enabled '
                              'else None uses first available GPU [Default=None]')
    options.add_argument('-l', '--loss', type=str, default=None, choices=('mse','mae','ncc','zncc'),
                         help='Use this specified loss function [Default=None, MSE for Unet]')
    options.add_argument('-lrs', '--lr-scheduler', action='store_true', default=False,
                         help='use a cosine-annealing based learning rate scheduler [Default=False]')
    options.add_argument('-mg', '--multi-gpu', action='store_true', default=False, help='use multiple gpus [Default=False]')
    options.add_argument('-n', '--n-jobs', type=int, default=0,
                            help='number of CPU processors to use (use 0 if CUDA enabled) [Default=0]')
    options.add_argument('-nlo', '--no-load-opt', action='store_true', default=False,
                         help='if loading a trained model, do not load the optimizer [Default=False]')
    options.add_argument('-ocf', '--out-config-file', type=str, default=None,
                         help='output a config file for the options used in this experiment '
                              '(saves them as a json file with the name as input in this argument)')
    options.add_argument('-ps', '--patch-size', type=int, default=64,
                         help='patch size^3 extracted from image [Default=64]')
    options.add_argument('-pm','--pin-memory', action='store_true', default=False, help='pin memory in dataloader [Default=False]')
    options.add_argument('-pl', '--plot-loss', type=str, default=None,
                            help='plot the loss vs epoch and save at the filename provided here [Default=None]')
    options.add_argument('-sa', '--sample-axis', type=int, default=2,
                            help='axis on which to sample for 2d (None for random orientation when NIfTI images given) [Default=2]')
    options.add_argument('-sd', '--seed', type=int, default=0, help='set seed for reproducibility [Default=0]')
    options.add_argument('-vs', '--valid-split', type=float, default=0.2,
                          help='split the data in source_dir and target_dir into train/validation '
                               'with this split percentage [Default=0]')
    options.add_argument('-vsd', '--valid-source-dir', type=str, default=None, nargs='+',
                          help='path to directory with source images for validation, '
                               'see -vs for default action if this is not provided [Default=None]')
    options.add_argument('-vtd', '--valid-target-dir', type=str, default=None, nargs='+',
                          help='path to directory with target images for validation, '
                               'see -vs for default action if this is not provided [Default=None]')
    options.add_argument('-v', '--verbosity', action="count", default=0,
                         help="increase output verbosity (e.g., -vv is more than -v)")

    nn_options = parser.add_argument_group('Neural Network Options')
    nn_options.add_argument('-ac', '--activation', type=str, default='relu', choices=('relu', 'lrelu'),
                            help='type of activation to use throughout network except output [Default=relu]')
    nn_options.add_argument('-atu', '--add-two-up', action='store_true', default=False,
                            help='Add two to the kernel size on the upsampling in the U-Net as '
                                 'per Zhao, et al. 2017 [Default=False]')
    nn_options.add_argument('-at', '--attention', action='store_true', default=False,
                            help='use attention gates in up conv layers in unet [Default=False]')
    nn_options.add_argument('-cbp', '--channel-base-power', type=int, default=5,
                            help='2 ** channel_base_power is the number of channels in the first layer '
                                 'and increases in each proceeding layer such that in the n-th layer there are '
                                 '2 ** (channel_base_power + n) channels [Default=5]')
    nn_options.add_argument('-dp', '--dropout-prob', type=float, default=0,
                            help='dropout probability per conv block [Default=0]')
    nn_options.add_argument('-eb', '--enable-bias', action='store_true', default=False,
                            help='enable bias calculation in upsampconv layers and final conv layer [Default=False]')
    nn_options.add_argument('-in', '--init', type=str, default='kaiming', choices=('normal', 'xavier', 'kaiming', 'orthogonal'),
                            help='use this type of initialization for the network [Default=kaiming]')
    nn_options.add_argument('-ing', '--init-gain', type=float, default=0.2,
                            help='use this initialization gain for initialization [Default=0.2]')
    nn_options.add_argument('-im', '--interp-mode', type=str, default='nearest', choices=('nearest','bilinear','trilinear'),
                            help='use this type of interpolation for upsampling [Default=nearest]')
    nn_options.add_argument('-ks', '--kernel-size', type=int, default=3,
                            help='convolutional kernel size (cubed) [Default=3]')
    nn_options.add_argument('-lr', '--learning-rate', type=float, default=1e-3,
                            help='learning rate of the neural network (uses Adam) [Default=1e-3]')
    nn_options.add_argument('-ne', '--n-epochs', type=int, default=100,
                            help='number of epochs [Default=100]')
    nn_options.add_argument('-nl', '--n-layers', type=int, default=3,
                            help='number of layers to use in network (different meaning per arch) [Default=3]')
    nn_options.add_argument('-3d', '--net3d', action='store_true', default=False, help='create a 3d network instead of 2d [Default=False]')
    nn_options.add_argument('-na', '--nn-arch', type=str, default='unet', choices=('unet', 'nconv', 'vae'),
                            help='specify neural network architecture to use')
    nn_options.add_argument('-ns', '--no-skip', action='store_true', default=False, help='do not use skip connections in unet [Default=False]')
    nn_options.add_argument('-nz', '--noise-lvl', type=float, default=0, help='add this level of noise to model parameters [Default=0]')
    nn_options.add_argument('-nm', '--normalization', type=str, default='instance',
                            choices=('instance', 'batch', 'layer', 'weight', 'spectral', 'none'),
                            help='type of normalization layer to use in network [Default=instance]')
    nn_options.add_argument('-ord', '--ord-params', type=int, nargs=3, default=None,
                            help='ordinal regression params (start, stop, n_bins) [Default=None]')
    nn_options.add_argument('-oac', '--out-activation', type=str, default='linear', choices=('relu', 'lrelu', 'linear'),
                            help='type of activation to use in network on output [Default=linear]')

    vae_options = parser.add_argument_group('VAE Options')
    vae_options.add_argument('-id', '--img-dim', type=int, nargs='+', default=None, help='if using VAE, then input image dimension must '
                                                                                  'be specified [Default=None]')
    vae_options.add_argument('-ls', '--latent-size', type=int, default=2048, help='if using VAE, this controls latent dimension size [Default=2048]')

    aug_options = parser.add_argument_group('Data Augmentation Options')
    aug_options.add_argument('-p', '--prob', type=float, nargs=5, default=None, help='probability of (Affine, Flip, Gamma, Block, Noise) [Default=None]')
    aug_options.add_argument('-r', '--rotate', type=float, default=0, help='max rotation angle [Default=0]')
    aug_options.add_argument('-ts', '--translate', type=float, default=None, help='max fractional translation [Default=None]')
    aug_options.add_argument('-sc', '--scale', type=float, default=None, help='max scale (1-scale,1+scale) [Default=None]')
    aug_options.add_argument('-hf', '--hflip', action='store_true', default=False, help='horizontal flip [Default=False]')
    aug_options.add_argument('-vf', '--vflip', action='store_true', default=False, help='vertical flip [Default=False]')
    aug_options.add_argument('-g', '--gamma', type=float, default=None, help='gamma (1-gamma,1+gamma) for (gain * x ** gamma) [Default=None]')
    aug_options.add_argument('-gn', '--gain', type=float, default=None, help='gain (1-gain,1+gain) for (gain * x ** gamma) [Default=None]')
    aug_options.add_argument('-blk', '--block', type=int, nargs=2, default=None, help='insert random blocks of this size range [Default=None]')
    aug_options.add_argument('-std', '--noise-std', type=float, default=0, help='noise standard deviation/power [Default=0]')
    aug_options.add_argument('-tx', '--tfm-x', action='store_true', default=True, help='apply transforms to x (change this with config file) [Default=True]')
    aug_options.add_argument('-ty', '--tfm-y', action='store_true', default=False, help='apply transforms to y [Default=False]')
    return parser


def criterion(out, tgt, model):
    """ helper function to handle multiple outputs in model evaluation """
    if isinstance(out, tuple):
        loss = model.module.criterion(tgt, *out) if isinstance(model, nn.DataParallel) else model.criterion(tgt, *out)
    else:
        loss = model.module.criterion(tgt, out) if isinstance(model, nn.DataParallel) else model.criterion(tgt, out)
    return loss


def load_model(model, fn, device):
    checkpoint = torch.load(fn)
    start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device)
    return model, start_epoch


def load_opt(optimizer, fn):
    checkpoint = torch.load(fn)
    optimizer.load_state_dict(checkpoint['optimizer'])
    return optimizer


def save_model(model, optimizer, t, fn):
    state_dict = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
    state = {'epoch': t, 'state_dict': state_dict, 'optimizer': optimizer.state_dict()}
    torch.save(state, fn)


######### Main routine ###########

def main(args=None):
    args, no_config_file = get_args(args, arg_parser)
    setup_log(args.verbosity)
    logger = logging.getLogger(__name__)
    try:
        # set random seeds for reproducibility
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        # define device to put tensors on
        device, use_cuda, n_gpus = get_device(args, logger)

        # import and initialize mixed precision training package
        amp_handle = None
        if args.fp16:
            try:
                from apex import amp
                amp_handle = amp.init()
            except ImportError:
                logger.info('Mixed precision training (i.e., the package `apex`) not available.')

        use_3d = args.net3d and args.ext is None
        if args.net3d and args.ext is not None: logger.warning(f'Cannot train a 3D network with {args.ext} images, creating a 2D network.')
        n_input, n_output = len(args.source_dir), len(args.target_dir)

        if args.ord_params is not None and n_output > 1:
            raise SynthNNError('Ordinal regression does not support multiple outputs.')

        if args.attention and args.net3d:
            logger.warning('Cannot use attention with 3D networks, not using attention.')
            args.attention = False

        # get the desired neural network architecture
        if args.nn_arch == 'nconv':
            from synthnn.models.nconvnet import SimpleConvNet
            logger.warning('The nconv network is for basic testing.')
            model = SimpleConvNet(args.n_layers, kernel_size=args.kernel_size, dropout_p=args.dropout_prob,
                                  n_input=n_input, n_output=n_output, is_3d=use_3d)
        elif args.nn_arch == 'unet':
            from synthnn.models.unet import Unet
            model = Unet(args.n_layers, kernel_size=args.kernel_size, dropout_p=args.dropout_prob,
                         channel_base_power=args.channel_base_power, add_two_up=args.add_two_up, normalization=args.normalization,
                         activation=args.activation, output_activation=args.out_activation, interp_mode=args.interp_mode,
                         enable_dropout=True, enable_bias=args.enable_bias, is_3d=use_3d,
                         n_input=n_input, n_output=n_output, no_skip=args.no_skip,
                         ord_params=args.ord_params, noise_lvl=args.noise_lvl, device=device,
                         loss=args.loss, attention=args.attention)
        elif args.nn_arch == 'vae':
            from synthnn.models.vae import VAE
            model = VAE(args.n_layers, args.img_dim, channel_base_power=args.channel_base_power, activation=args.activation,
                        is_3d=use_3d, n_input=n_input, n_output=n_output, latent_size=args.latent_size)
        else:
            raise SynthNNError(f'Invalid NN type: {args.nn_arch}. {{nconv, unet, vae}} are the only supported options.')
        model.train(True)
        logger.debug(model)
        logger.info(f'Number of trainable parameters in model: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')

        # load a trained model if desired
        if os.path.isfile(args.trained_model):
            model, start_epoch = load_model(model, args.trained_model, device)
            model.train()
            logger.info(f'Loaded checkpoint: {args.trained_model} (epoch {start_epoch})')

        # put the model on the GPU if available and desired
        if use_cuda: model.cuda(device=device)
        use_multi = args.multi_gpu and n_gpus > 1 and use_cuda
        if args.multi_gpu and n_gpus <= 1: logger.warning('Multi-GPU functionality is not available on your system.')
        if use_multi:
            n_gpus = len(args.gpu_selector) if args.gpu_selector is not None else n_gpus
            logger.info(f'Enabling use of {n_gpus} gpus')
            model = torch.nn.DataParallel(model, device_ids=args.gpu_selector)

        # initialize/load optimizer state
        logger.info(('Max ' if args.lr_scheduler else '') + f'LR: {args.learning_rate:.5f}')
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        if os.path.isfile(args.trained_model) and not args.no_load_opt:
            optimizer = load_opt(optimizer, args.trained_model)

        # initialize the weights with user-defined initialization routine
        logger.info(f'Initializing weights with {args.init}')
        init_weights(model, args.init, args.init_gain)

        # check number of jobs requested and CPUs available
        num_cpus = os.cpu_count()
        if num_cpus < args.n_jobs:
            logger.warning(f'Requested more workers than available (n_jobs={args.n_jobs}, # cpus={num_cpus}). '
                           f'Setting n_jobs={num_cpus}.')
            args.n_jobs = num_cpus

        # control random cropping patch size (or if used at all)
        if args.ext is None:
            cropper = tfms.RandomCrop3D(args.patch_size) if args.net3d else tfms.RandomCrop2D(args.patch_size, args.sample_axis)
            tfm = [cropper] if args.patch_size > 0 else [] if args.net3d else [tfms.RandomSlice(args.sample_axis)]
        else:
            tfm = [tfms.RandomCrop(args.patch_size)] if args.patch_size > 0 else []

        # add data augmentation if desired
        if args.prob is not None:
            logger.info('Adding data augmentation transforms')
            if args.net3d and (args.prob[0] > 0 or args.prob[1] > 0 or args.prob[3] > 0):
                logger.warning('Cannot do affine or flipping or block data augmentation with 3d networks')
                args.prob[0], args.prob[1], args.prob[3] = 0, 0, 0
                args.rotate, args.translate, args.scale, args.hflip, args.vflip, args.block = 0, None, None, False, False, None
            tfm.extend(tfms.get_transforms(args.prob, args.tfm_x, args.tfm_y, args.rotate, args.translate, args.scale,
                                           args.vflip, args.hflip, args.gamma, args.gain, args.noise_std, args.block))
        else:
            logger.info('No data augmentation will be used (except random cropping if patch_size > 0)')
            tfm.append(tfms.ToTensor())

        # define dataset and split into training/validation set
        dataset = MultimodalNiftiDataset(args.source_dir, args.target_dir, Compose(tfm)) if args.ext is None else \
                  MultimodalImageDataset(args.source_dir, args.target_dir, Compose(tfm), ext='*.' + args.ext)
        logger.info(f'Number of training images: {len(dataset)}')

        if args.valid_source_dir is not None and args.valid_target_dir is not None:
            valid_dataset = MultimodalNiftiDataset(args.valid_source_dir, args.valid_target_dir, Compose(tfm)) if args.ext is None else \
                            MultimodalImageDataset(args.valid_source_dir, args.valid_target_dir, Compose(tfm), ext='*.' + args.ext)
            logger.info(f'Number of validation images: {len(valid_dataset)}')
            train_loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.n_jobs, shuffle=True, pin_memory=args.pin_memory)
            validation_loader = DataLoader(valid_dataset, batch_size=args.batch_size, num_workers=args.n_jobs, pin_memory=args.pin_memory)
        else:
            # setup training and validation set
            num_train = len(dataset)
            indices = list(range(num_train))
            split = int(args.valid_split * num_train)
            validation_idx = np.random.choice(indices, size=split, replace=False)
            train_idx = list(set(indices) - set(validation_idx))

            train_sampler = SubsetRandomSampler(train_idx)
            validation_sampler = SubsetRandomSampler(validation_idx)

            # set up data loader for nifti images
            train_loader = DataLoader(dataset, sampler=train_sampler, batch_size=args.batch_size, num_workers=args.n_jobs, pin_memory=args.pin_memory)
            validation_loader = DataLoader(dataset, sampler=validation_sampler, batch_size=args.batch_size, num_workers=args.n_jobs, pin_memory=args.pin_memory)

        # setup a learning rate scheduler if desired
        if args.lr_scheduler:
            logger.info('Enabling burn-in cosine annealing LR scheduler')
            scheduler = BurnCosineLR(optimizer, args.n_epochs)

        # training and validation loop
        use_valid = args.valid_split > 0 or (args.valid_source_dir is not None and args.valid_target_dir is not None)
        train_losses, validation_losses = [], []
        for t in range(1, args.n_epochs+1):
            # training
            t_losses = []
            if use_valid: model.train(True)
            for src, tgt in train_loader:
                src, tgt = src.to(device), tgt.to(device)
                out = model(src)
                loss = criterion(out, tgt, model)
                t_losses.append(loss.item())
                optimizer.zero_grad()
                if args.fp16 and amp_handle is not None:
                    with amp_handle.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()
                if args.clip is not None: nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                optimizer.step()
            train_losses.append(t_losses)
            if args.lr_scheduler: scheduler.step()

            if args.checkpoint is not None:
                if t % args.checkpoint == 0:
                    path, base, ext = split_filename(args.trained_model)
                    fn = os.path.join(path, base + f'_chk_{t}' + ext)
                    save_model(model, optimizer, t, fn)

            # validation
            v_losses = []
            if use_valid: model.train(False)
            with torch.set_grad_enabled(False):
                for src, tgt in validation_loader:
                    src, tgt = src.to(device), tgt.to(device)
                    out = model(src)
                    loss = criterion(out, tgt, model)
                    v_losses.append(loss.item())
                validation_losses.append(v_losses)

            if np.any(np.isnan(t_losses)): raise SynthNNError('NaN in training loss, cannot recover. Exiting.')
            log = f'Epoch: {t} - Training Loss: {np.mean(t_losses):.2e}'
            if use_valid: log += f', Validation Loss: {np.mean(v_losses):.2e}'
            if args.lr_scheduler: log += f', LR: {scheduler.get_lr()[0]:.2e}'
            logger.info(log)

        # output a config file if desired
        if args.out_config_file is not None:
            write_out_config(args, n_gpus, n_input, n_output, use_3d)

        # save the trained model
        use_config_file = not no_config_file or args.out_config_file is not None
        if use_config_file:
            save_model(model, optimizer, args.n_epochs, args.trained_model)
        else:
            # save the whole model (if changes occur to pytorch, then this model will probably not be loadable)
            logger.warning('Saving the entire model. Preferred to create a config file and only save model weights')
            torch.save(model, args.trained_model)

        # plot the loss vs epoch (if desired)
        if args.plot_loss is not None:
            plot_error = True if args.n_epochs <= 30 else False
            from synthnn import plot_loss
            if matplotlib.get_backend() != 'agg':
                import matplotlib.pyplot as plt
                plt.switch_backend('agg')
            ax = plot_loss(train_losses, ecolor='darkorchid', label='Train', plot_error=plot_error)
            _ = plot_loss(validation_losses, filename=args.plot_loss, ecolor='firebrick', ax=ax, label='Validation', plot_error=plot_error)

        return 0
    except Exception as e:
        logger.exception(e)
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
