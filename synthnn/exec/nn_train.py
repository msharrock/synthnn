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
import logging
import sys
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', category=UserWarning)
    import matplotlib
    matplotlib.use('agg')  # do not pull in GUI
    import numpy as np
    import torch
    from synthnn import Learner
    from .exec import get_args, setup_log


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
    options.add_argument('-opt', '--optimizer', type=str, default='adam',
                         choices=('adam','sgd','adagrad','amsgrad','rmsprop','adabound','amsbound'),
                         help='Use this optimizer to train the network [Default=adam]')
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
                               'with this split percentage [Default=0.2]')
    options.add_argument('-vsd', '--valid-source-dir', type=str, default=None, nargs='+',
                          help='path to directory with source images for validation, '
                               'see -vs for default action if this is not provided [Default=None]')
    options.add_argument('-vtd', '--valid-target-dir', type=str, default=None, nargs='+',
                          help='path to directory with target images for validation, '
                               'see -vs for default action if this is not provided [Default=None]')
    options.add_argument('-v', '--verbosity', action="count", default=0,
                         help="increase output verbosity (e.g., -vv is more than -v)")

    nn_options = parser.add_argument_group('Neural Network Options')
    nn_options.add_argument('-ac', '--activation', type=str, default='relu',
                            choices=('relu', 'lrelu','prelu','elu','celu','selu','tanh','sigmoid'),
                            help='type of activation to use throughout network except output [Default=relu]')
    nn_options.add_argument('-atu', '--add-two-up', action='store_true', default=False,
                            help='Add two to the kernel size on the upsampling in the U-Net as '
                                 'per Zhao, et al. 2017 [Default=False]')
    nn_options.add_argument('-at', '--attention', action='store_true', default=False,
                            help='use attention gates in up conv layers in unet[Default=False]')
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
    nn_options.add_argument('-oac', '--out-activation', type=str, default='linear',
                            choices=('linear','relu', 'lrelu','prelu','elu','celu','selu','tanh','sigmoid'),
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


######### Main routine ###########

def main(args=None):
    args, no_config_file = get_args(args, arg_parser)
    setup_log(args.verbosity)
    logger = logging.getLogger(__name__)
    try:
        # set random seeds for reproducibility
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        learner = Learner.train_setup(args)

        if args.fp16: learner.fp16()
        if args.multi_gpu: learner.multigpu(args.gpu_selector)
        if args.lr_scheduler: learner.lr_scheduler(args.n_epochs)

        train_loss, valid_loss = learner.fit(args.n_epochs, args.clip, args.checkpoint, args.trained_model)

        # output a config file if desired
        if args.out_config_file is not None: args.write_json(args.out_config_file)

        # save the trained model
        learner.save(args.trained_model, args.n_epochs)

        # plot the loss vs epoch (if desired)
        if args.plot_loss is not None: learner.plot_loss(train_loss, valid_loss, args.plot_loss)

        return 0
    except Exception as e:
        logger.exception(e)
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
