import os
from typing import Any, Dict, Tuple, List
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import math
import random
import imgaug
import json
from ..utils.load_save import read_img
from .utils import *
from .augmentor import augmentor


class HVDataset(Dataset):
    def __init__(self, cfg, mode):
        """
        Parameters
        ----------
        images: list of path of images
        labels: list of path of labels
        mode: 'train' or 'test'
        input_shape:   shape required to input to the net H*W
        """
        self.cfg = cfg
        dataset = cfg.DATASET.INPUTPATH + cfg.DATASET.IMAGE if mode == 'train' else cfg.DATASET.INPUTPATH + cfg.INFERENCE.IMAGE
        self.images = json.load(dataset)['image']
        self.labels = json.load(dataset)['label'] if mode == 'train' else None
        self.input_path = cfg.DATASET.INPUTPATH
        self.mode = mode
        self.input_shape = cfg.MODEL.INPUT_SHAPE
        self.output_shape = cfg.MODEL.OUTPUT_SHAPE
        self.reject_size = cfg.MODEL.REJECT_SIZE
        self.reject_p = cfg.MODEL.REJECT_P
        self.stride = cfg.INFERENCE.STRIDE
        self.augmentor = augmentor(cfg)
        assert os.path.isdir(self.input_path)
        if mode == 'test':
            # at test mode, first load all images, and compute the position
            self.pos = []
            self.shapes = []
            for idx in range(len(self.images)):
                image = read_img(os.path.join(self.input_path, self.images[idx]))
                self.pos.extend(self.compute_pos(image.shape, idx))
                self.shapes.append(image.shape[:2])

    def __getitem__(self, idx: int):
        if self.mode == 'train':
            """
            read image -> pad image -> sample image -> aug 
            """
            image = read_img(os.path.join(self.input_path, self.images[idx]))
            mask = read_img(os.path.join(self.input_path, self.labels[idx]))
            image, mask = self.pad(image, mask)
            pos = self.compute_pos(image.shape)
            image, mask = self.sample(pos, image, mask)
            image, mask = self.augmentor({'image': image, 'label': mask})
            return image, mask
        if self.mode == 'test':
            """
            read image -> compute pos
            """
            pos = self.pos[idx]
            image = read_img(os.path.join(self.input_path, self.images[pos[0]]))
            image = self.pad(image)
            image = self.sample(pos[1:], image)
            return image, pos

    def __len__(self):
        if self.mode == 'train':
            return len(self.images)
        if self.mode == 'test':
            return len(self.pos)

    def pad(self, image, mask=None):
        # N*stride + input_shape <= image shape < (N+1)*stride + input_shape
        a = [0, 0]
        b = [0, 0]
        for i in range(2):
            N = max(0, (image.shape[i] - self.output_shape[i])) // self.stride
            if image.shape[i] > (N * self.stride + self.output_shape[i]):
                b[i] = image.shape[i] - (N * self.stride + self.output_shape[i])

        a[0] += (self.input_shape[0] - self.output_shape[0]) // 2
        b[0] += (self.input_shape[0] - self.output_shape[0]) // 2
        a[1] += (self.input_shape[1] - self.output_shape[1]) // 2
        b[1] += (self.input_shape[1] - self.output_shape[1]) // 2

        image0 = np.squeeze(image[..., 0])
        image1 = np.squeeze(image[..., 1])
        image2 = np.squeeze(image[..., 2])
        bv = int(np.mean(image[:, 0, 0]))
        gv = int(np.mean(image[:, 0, 1]))
        rv = int(np.mean(image[:, 0, 2]))
        # image = np.lib.pad(image, ((a1 + 64, b1 + 64), (a2 + 64, b2 + 64), (0, 0)), mode='reflect')
        image0 = np.pad(image0, ((a[0], b[0]), (a[1], b[1])), constant_values=(bv, bv), mode='constant')
        image1 = np.pad(image1, ((a[0], b[0]), (a[1], b[1])), constant_values=(gv, gv), mode='constant')
        image2 = np.pad(image2, ((a[0], b[0]), (a[1], b[1])), constant_values=(rv, rv), mode='constant')
        image0 = np.expand_dims(image0, axis=2)
        image1 = np.expand_dims(image1, axis=2)
        image2 = np.expand_dims(image2, axis=2)
        image = np.concatenate([image0, image1, image2], axis=2)
        if self.mode == 'train':
            mask = np.pad(mask, ((a[0], b[0]), (a[1], b[1]), (0, 0)), constant_values=0, mode='constant')
            return image, mask
        else:
            return image

    def compute_pos(self, image_shape, idx=None):
        if self.mode == 'test':
            # only pad at the end
            pos = [[], []]
            for i in range(2):
                N = max(0, (image_shape[i] - self.output_shape[i])) // self.stride
                if image_shape[i] > (N * self.stride + self.output_shape[i]):
                    N += 1
                for j in range(N):
                    pos[i].append(j * self.stride)
            if idx is None:
                pos = [[pos[0][i], pos[1][i]] for i in range(len(pos[0]))]
                return pos
            else:
                pos = [[idx, pos[0][i], pos[1][i]] for i in range(len(pos[0]))]
                return pos
        if self.mode == 'train':
            return [random.randint(0, image_shape[i] - self.input_shape[i]) for i in range(2)]

    def sample(self, pos, image, mask=None):
        if self.mode == 'train':
            new_image = image[pos[0]:pos[0] + self.input_shape[0], pos[1]:pos[1] + self.input_shape[1], :]
            new_mask = mask[pos[0]:pos[0] + self.input_shape[0], pos[1]:pos[1] + self.input_shape[1], :]
            return new_image, new_mask
        if self.mode == 'test':
            new_image = image[pos[0]:pos[0] + self.input_shape[0], pos[1]:pos[1] + self.input_shape[1], :]
            return new_image
