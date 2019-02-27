#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
synthnn.learn.predict

routines specific to prediction

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)

Created on: Feb 26, 2018
"""

__all__ = ['Predictor']

import logging
import math
import warnings

import numpy as np
import torch

logger = logging.getLogger(__name__)


class Predictor:

    def __init__(self, model:torch.nn.Module, patch_size:int, batch_size:int, device:torch.device,
                 axis:int=0, n_output:int=1, is_3d:bool=False):
        self.model = model
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.device = device
        self.axis = axis
        self.n_output = n_output
        self.is_3d = is_3d

    def predict(self, img:np.ndarray, nsyn:int=1, temperature_map:bool=False, calc_var:bool=False) -> np.ndarray:
        """ picks and runs the correct prediction routine based on input info """
        if self.patch_size > 0 and self.is_3d:
            out_img = self.patch_3d_predict(img, nsyn, temperature_map, calc_var)
        elif self.is_3d:
            out_img = self.whole_3d_predict(img, nsyn, temperature_map, calc_var)
        else:
            out_img = self.slice_predict(img, nsyn, temperature_map, calc_var)
        return out_img

    def whole_3d_predict(self, img:np.ndarray, nsyn:int=1, temperature_map:bool=False, calc_var:bool=False) -> np.ndarray:
        """ 3d whole-image-based prediction """
        if img.ndim == 3: img = img[np.newaxis, ...]
        out_img = np.zeros((nsyn,) + img.shape)
        test_img = torch.from_numpy(img).to(self.device)[None, ...]  # add empty batch dimension
        for j in range(nsyn):
            out_img[j] = self._fwd(test_img, temperature_map)[0]  # remove empty batch dimension
        out_img = np.mean(out_img, axis=0) if not calc_var else np.var(out_img, axis=0)
        return out_img

    def patch_3d_predict(self, img:np.ndarray, nsyn:int=1, temperature_map:bool=False, calc_var:bool=False) -> np.ndarray:
        """ 3d patch-by-patch based prediction """
        if img.ndim == 3: img = img[np.newaxis, ...]
        out_img = np.zeros((self.n_output,) + img.shape[1:])
        count_mtx = np.zeros(img.shape[1:])
        x, y, z = self._get_overlapping_3d_idxs(self.patch_size, img)
        dec_idxs = np.floor(np.percentile(np.arange(x.shape[0]), np.arange(0, 101, 5)))
        pct_complete = 0
        j = 0
        # The below for-loop handles collecting overlapping patches and putting
        # them into a batch format that pytorch models expect (i.e., [N,C,H,W,D])
        # and running the batch through the network (storing the results in out_img).
        for i, (xx, yy, zz) in enumerate(zip(x, y, z)):
            if i in dec_idxs:
                logger.info(f'{pct_complete}% Complete')
                pct_complete += 5
            count_mtx[xx, yy, zz] = count_mtx[xx, yy, zz] + 1
            if j == 0:
                batch = np.zeros((self.batch_size,) + img[:, xx, yy, zz].shape, dtype=np.float32)
                batch_idxs = [(xx, yy, zz)]
                batch[j, ...] = img[:, xx, yy, zz]
                j += 1
            elif j != self.batch_size:
                batch_idxs.append((xx, yy, zz))
                batch[j, ...] = img[:, xx, yy, zz]
                j += 1
            else:
                batch = torch.from_numpy(batch).to(self.device)
                predicted = np.zeros(batch.shape)
                for _ in range(nsyn):
                    predicted += self._fwd(batch, temperature_map)
                for ii, (bx, by, bz) in enumerate(batch_idxs):
                    out_img[:, bx, by, bz] = out_img[:, bx, by, bz] + predicted[ii, ...]
                j = 0
        count_mtx[count_mtx == 0] = 1  # avoid division by zero
        out_img /= count_mtx
        return out_img

    def slice_predict(self, img:np.ndarray, nsyn:int=1, temperature_map:bool=False, calc_var:bool=False) -> np.ndarray:
        """ slice-by-slice based prediction """
        if img.ndim == 2: img = img[np.newaxis, ...]
        out_img = np.zeros((nsyn, self.n_output) + img.shape[1:])
        num_batches = math.floor(img.shape[self.axis + 1] / self.batch_size)  # add one to axis to ignore channel dim
        if img.shape[self.axis + 1] / self.batch_size != num_batches:
            lbi = int(num_batches * self.batch_size)  # last batch index
            num_batches += 1
            lbs = img.shape[self.axis + 1] - lbi  # last batch size
        else:
            lbi = None
        for i in range(num_batches if lbi is None else num_batches - 1):
            logger.info(f'Starting batch ({i + 1}/{num_batches})')
            self._batch2d(img, out_img, i * self.batch_size, nsyn, temperature_map)
        if lbi is not None:
            logger.info(f'Starting batch ({num_batches}/{num_batches})')
            self._batch2d(img, out_img, lbi, nsyn, temperature_map, lbs)
        out_img = np.mean(out_img, axis=0) if not calc_var else np.var(out_img, axis=0)
        return out_img

    def _fwd(self, img, temperature_map):
        return self.model.predict(img, temperature_map).cpu().detach().numpy()

    @staticmethod
    def _get_overlapping_3d_idxs(psz, img):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            stride = psz // 2
            indices = [torch.from_numpy(idxs) for idxs in np.indices(img.shape[1:])]
            for i in range(3):  # create blocks from imgs (and indices)
                indices = [idxs.unfold(i, psz, stride) for idxs in indices]
            x, y, z = [idxs.contiguous().view(-1, psz, psz, psz) for idxs in indices]
        return x, y, z

    def _batch2d(self, img, out_img, i, nsyn, temperature_map, bs=None):
        bs = bs or self.batch_size
        s = np.transpose(img[:,i:i+bs,:,:],[1,0,2,3]) if self.axis == 0 else \
            np.transpose(img[:,:,i:i+bs,:],[2,0,1,3]) if self.axis == 1 else \
            np.transpose(img[:,:,:,i:i+bs],[3,0,1,2])
        img_b = torch.from_numpy(s).to(self.device)
        for j in range(nsyn):
            x = self._fwd(img_b, temperature_map)
            if self.axis == 0:
                out_img[j,:,i:i+bs,:,:] = np.transpose(x, [1,0,2,3])
            elif self.axis == 1:
                out_img[j,:,:,i:i+bs,:] = np.transpose(x, [1,2,0,3])
            else:
                out_img[j,:,:,:,i:i+bs] = np.transpose(x, [1,2,3,0])
