import numpy as np
import imageio
import os
import torch
from scipy.io import savemat
import h5py


def readh5(filename, dataset=''):
    fid = h5py.File(filename, 'r')
    if dataset == '':
        dataset = list(fid)[0]
    return np.array(fid[dataset])


def read_img(path):
    suf = path.split('.')[-1]
    if suf == 'npy':
        image = np.load(path, allow_pickle=True)
    elif suf == 'tif':
        image = imageio.imread(path)
    elif suf == 'h5':
        image = readh5(path)
    else:
        raise ValueError('unacceptable image format')
    return image


def save_img(path, image, mode='npy'):
    if mode == 'npy':
        np.save(path, image)
    if mode == 'mat':
        pass
