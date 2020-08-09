import os
from typing import Any, Dict, Tuple, List
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import math
import random
import imgaug

from ..utils.load_save import read_img
from .utils import *


class HVDataset(Dataset):
    def __init__(self,
                 images: List,
                 labels: List,
                 mode: str,
                 input_shape,
                 output_shape,
                 input_path: str,
                 reject_size: int,
                 reject_p: float,
                 stride: List[int, int]):
        """
        Parameters
        ----------
        images: list of path of images
        labels: list of path of labels
        mode: 'train' or 'test'
        input_shape:   shape required to input to the net H*W
        """
        self.input_path = input_path
        self.input_shape = input_shape
        self.output_shape = output_shape
        assert os.path.isdir(self.input_path)
        self.images = images
        self.labels = labels if mode == 'train' else None
        self.mode = mode
        self.reject_size = reject_size
        self.reject_p = reject_p
        # judge if the current image has been all
        # predicted in "test" phase
        # image_idx, pos_idx, pos_max_idx
        self.index = [0, 0, 0]
        self.stride = stride

    def __getitem__(self, idx: int):
        if self.mode == 'train':
            self.image = read_img(os.path.join(self.input_path, self.images[idx]))
            self.mask = read_img(os.path.join(self.input_path, self.labels[idx]))
            self.image, self.mask = self.pad(self.image, self.mask)
            image, mask = self.sample()
            image, mask = aug(image, mask)
            return image, mask, self.images[idx]
        else:
            if self.index[1] == self.index[2]:
                self.index[0] += 1
                self.index[1] = self.index[2] = 0
                self.image = read_img(os.path.join(self.input_path, self.images[self.index[0]]))
                self.image = self.pad(self.image)
            elif self.index[1] < self.index[2]:
                self.index[1] += 1
            image = self.sample()
            return image, self.index, self.images[idx]

    def __len__(self):
        return len(self.images)

    def sample(self):
        shape = self.mask.shape[:2]
        assert shape[0] >= self.input_shape[0] and shape[1] >= self.input_shape[1]
        if self.mode == 'train':
            while True:
                # H W C
                pos = [random.randint(0, shape[i] - self.input_shape[i]) for i in range(2)]
                mask = self.mask[
                       pos[0] + self.input_shape[0] // 2 - self.output_shape[0] // 2:
                       pos[0] + self.input_shape[0] // 2 + self.output_shape[0] // 2,
                       pos[1] + self.input_shape[1] // 2 - self.output_shape[1] // 2:
                       pos[1] + self.input_shape[1] // 2 + self.output_shape[1] // 2, :]
                if (mask[0] > 0).sum() > self.reject_size:
                    image = self.image[pos[0]:pos[0] + self.input_shape[0], pos[1]:pos[1] + self.input_shape[1], :]
                    return image, mask
                elif random.random() > self.reject_p:
                    image = self.image[pos[0]:pos[0] + self.input_shape[0], pos[1]:pos[1] + self.input_shape[1], :]
                    return image, mask
        else:
            t = [(self.image.shape[i] - self.input_shape[i]) // self.stride[i] for i in range(2)]
            pos = [0, 0]
            pos[0] = (self.index[0] // t[0]) * self.stride[0]
            pos[1] = (self.index[1] % t[1]) * self.stride[1]
            image = self.image[pos[0]:pos[0] + self.input_shape[0],
                    pos[1]:pos[1] + self.input_shape[1]]
            return image

    def pad(self, image, mask=None):
        # before, after
        # k*stride + output_shape = pad + image_shape
        pad = [[0, 0], [0, 0]]
        for i in range(2):
            k = 0
            while True:
                if k * self.stride[i] + self.output_shape[i] - image.shape[i] >= 0:
                    break
                else:
                    k += 1
            tmp = k * self.stride[i] + self.output_shape[i] - image.shape[i]
            pad[i][0] = tmp // 2
            pad[i][1] = tmp - tmp // 2
        # padding around edge
        dif = [self.input_shape[i] - self.output_shape[i] for i in range(2)]
        for i in image.shape[2]:
            pad_value = int(np.mean(image[:, 0, 0]))
            image[..., i] = np.pad(image[..., i],
                                   ((pad[0][0] + dif[0] // 2, pad[0][1] + dif[0] // 2),
                                    (pad[1][0] + dif[1] // 2, pad[1][1] + dif[1] // 2)),
                                   constant_values=(pad_value, pad_value),
                                   mode='constant')

        if mask is not None:
            mask = np.pad(mask, ((pad[0][0], pad[0][1]), (pad[1][0], pad[1][1]), (0, 0)), mode='constant')
            return image, mask
