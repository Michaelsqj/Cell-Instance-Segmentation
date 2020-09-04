import numpy as np
import imageio
import os
import torch


def read_img(path):
    suf = path.split('.')[-1]
    if suf == 'npy':
        image = np.load(path, allow_pickle=True)
    else:
        image = imageio.imread(path)
    return image


def save_img(path, image, mode='npy'):
    if mode == 'npy':
        np.save(path, image)
