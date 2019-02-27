#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
synthnn.util.config

create class for experiment configuration in the synthnn package

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)

Created on: Feb 26, 2018
"""

__all__ = ['ExperimentConfig']

import json
import logging

from ..errors import SynthNNError

logger = logging.getLogger(__name__)


class ExperimentConfig(dict):

    def __init__(self, *args, **kwargs):
        self.predict_dir = None
        self.predict_out = None
        self.source_dir = None
        self.target_dir = None
        self.trained_model = None
        self.batch_size = None
        self.disable_cuda = None
        self.ext = None
        self.gpu_selector = None
        self.multi_gpu = None
        self.out_config_file = None
        self.patch_size = None
        self.pin_memory = None
        self.sample_axis = None
        self.seed = None
        self.verbosity = None
        self.activation = None
        self.add_two_up = None
        self.attention = None
        self.channel_base_power = None
        self.dropout_prob = None
        self.enable_bias = None
        self.init = None
        self.init_gain = None
        self.interp_mode = None
        self.kernel_size = None
        self.n_layers = None
        self.net3d = None
        self.nn_arch = None
        self.no_skip = None
        self.noise_lvl = None
        self.normalization = None
        self.ord_params = None
        self.out_activation = None
        self.checkpoint = None
        self.clip = None
        self.fp16 = None
        self.learning_rate = None
        self.loss = None
        self.lr_scheduler = None
        self.n_epochs = None
        self.n_jobs = None
        self.no_load_opt = None
        self.optimizer = None
        self.plot_loss = None
        self.valid_source_dir = None
        self.valid_split = None
        self.valid_target_dir = None
        self.calc_var = None
        self.monte_carlo = None
        self.temperature_map = None
        self.img_dim = None
        self.latent_size = None
        self.n_gpus = None
        self.n_input = None
        self.n_output = None
        self.prob = None
        self.rotate = None
        self.translate = None
        self.scale = None
        self.hflip = None
        self.vflip = None
        self.gamma = None
        self.gain = None
        self.block = None
        self.noise_std = None
        self.tfm_x = None
        self.tfm_y = None
        super(ExperimentConfig, self).__init__(*args, **kwargs)
        self.__dict__ = self
        self._check_config()

    def _check_config(self):
        """ check to make sure requested configuration is valid """
        if self.ord_params is not None and self.n_output > 1:
            SynthNNError('Ordinal regression does not support multiple outputs.')

        if self.net3d and self.ext is not None:
            logger.warning(f'Cannot train a 3D network with {self.ext} images, creating a 2D network.')
            self.net3d = False

        if self.attention and self.net3d:
            logger.warning('Cannot use attention with 3D networks, not using attention.')
            self.attention = False

        if self.prob is not None:
            if self.net3d and (self.prob[0] > 0 or self.prob[1] > 0 or self.prob[3] > 0):
                logger.warning('Cannot do affine, flipping or block data augmentation with 3D networks.')
                self.prob[0], self.prob[1], self.prob[3] = 0, 0, 0
                self.rotate, self.translate, self.scale = 0, None, None
                self.hflip, self.vflip = False, False
                self.block = None

        if self.ord_params is None and self.temperature_map:
            logger.warning('temperature_map is only a valid option when using ordinal regression.')
            self.temperature_map = False

    @classmethod
    def load_json(cls, fn:str):
        """ handle loading from json file """
        with open(fn, 'r') as f:
            config = cls(_flatten(json.load(f)))  # dict comp. flattens first layer of dict
        return config

    @classmethod
    def from_argparse(cls, args):
        """ create an instance from a argument parser """
        args.n_gpus = 0
        args.n_input, args.n_output = len(args.source_dir), len(args.target_dir)
        arg_dict = _get_arg_dict(args)
        return cls(_flatten(arg_dict))

    def write_json(self, fn:str):
        """ write the experiment config to a file"""
        with open(fn, 'w') as f:
            arg_dict = _get_arg_dict(self.__dict__)
            json.dump(arg_dict, f, sort_keys=True, indent=2)


def _flatten(d): return {k: v for item in d.values() for k, v in item.items()}


def _get_arg_dict(args):
    arg_dict = {
        "Required": {
            "predict_dir": ["SET ME!"] if not hasattr(args,'predict_dir') else args.predict_dir,
            "predict_out": "SET ME!" if not hasattr(args,'predict_out') else args.predict_out,
            "source_dir": args.source_dir,
            "target_dir": args.target_dir,
            "trained_model": args.trained_model
        },
        "Options": {
            "batch_size": args.batch_size,
            "disable_cuda": args.disable_cuda,
            "ext": args.ext,
            "gpu_selector": args.gpu_selector,
            "multi_gpu": args.multi_gpu,
            "out_config_file": args.out_config_file,
            "patch_size": args.patch_size,
            "pin_memory": args.pin_memory,
            "sample_axis": args.sample_axis,
            "seed": args.seed,
            "verbosity": args.verbosity
        },
        "Neural Network Options": {
            "activation": args.activation,
            "add_two_up": args.add_two_up,
            "attention": args.attention,
            "channel_base_power": args.channel_base_power,
            "dropout_prob": args.dropout_prob,
            "enable_bias": args.enable_bias,
            "init": args.init,
            "init_gain": args.init_gain,
            "interp_mode": args.interp_mode,
            "kernel_size": args.kernel_size,
            "n_layers": args.n_layers,
            "net3d": args.net3d,
            "nn_arch": args.nn_arch,
            "no_skip": args.no_skip,
            "noise_lvl": args.noise_lvl,
            "normalization": args.normalization,
            "ord_params": args.ord_params,
            "out_activation": args.out_activation,
        },
        "Training Options": {
            "checkpoint": args.checkpoint,
            "clip": args.clip,
            "fp16": args.fp16,
            "learning_rate": args.learning_rate,
            "loss": args.loss,
            "lr_scheduler": args.lr_scheduler,
            "n_epochs": args.n_epochs,
            "n_jobs": args.n_jobs,
            "no_load_opt": args.no_load_opt,
            "optimizer": args.optimizer,
            "plot_loss": args.plot_loss,
            "valid_source_dir": args.valid_source_dir,
            "valid_split": args.valid_split,
            "valid_target_dir": args.valid_target_dir
        },
        "Prediction Options": {
            "calc_var": False if not hasattr(args,'calc_var') else args.calc_var,
            "monte_carlo": None if not hasattr(args,'monte_carlo') else args.monte_carlo,
            "temperature_map": False if not hasattr(args,'temperature_map') else args.temperature_map
        },
        "VAE Options": {
            "img_dim": args.img_dim,
            "latent_size": args.latent_size if args.nn_arch == 'vae' else None
        },
        "Internal": {
            "n_gpus": args.n_gpus,
            "n_input": args.n_input,
            "n_output": args.n_output
        },
        "Data Augmentation Options": {
            "prob": args.prob,
            "rotate": args.rotate,
            "translate": args.translate,
            "scale": args.scale,
            "hflip": args.hflip,
            "vflip": args.vflip,
            "gamma": args.gamma,
            "gain": args.gain,
            "block": args.block,
            "noise_std": args.noise_std,
            "tfm_x": args.tfm_x,
            "tfm_y": args.tfm_y
        }
    }
    return arg_dict
