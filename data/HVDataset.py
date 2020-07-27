import os
from typing import Any, Dict, Tuple, List
import torch
from torch.utils.data import DataLoader, Dataset
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
import numpy as np
import math
import random
import imgaug

from ..utils.load_save import read_img


class HVDataset(Dataset):
    def __init__(self,
                 images: List,
                 labels: List,
                 mode: str,
                 input_shape,
                 output_shape,
                 input_path: str
                 ):
        '''

        Parameters
        ----------
        images: list of path of images
        labels: list of path of labels
        mode: 'train' or 'test'
        input_shape:   shape required to input to the net H*W
        '''
        self.input_path = input_path
        self.input_shape = input_shape
        self.output_shape = output_shape
        assert os.path.isdir(self.input_path)
        self.images = images
        self.labels = labels if mode == 'train' else None
        self.mode = mode
        self.img_shape = [] * len(self.images)
        for i in range(len(self.images)):
            self.img_shape[i] = read_img(os.path.join(self.input_path, self.images[i])).shape

    def __getitem__(self, idx: int):
        image = read_img(os.path.join(self.input_path, self.images[idx]))
        if self.mode == 'train':
            mask = read_img(os.path.join(self.input_path, self.labels[idx]))
        image, mask = self.transform(image, mask)

        return image, mask, self.images[idx]

    def __len__(self):
        return len(self.images)

    def transform(self, image, mask):
        pad = [[0, 0], [0, 0]]
        for i in range(2):
            pad[i][0] = max(0, (self.output_shape[i] - image.shape[i]) // 2)
            pad[i][1] = max(0, self.output_shape[i] - image.shape[i] - (self.output_shape[i] - image.shape[i]) // 2)

        for i in image.shape[2]:
            pad_value = int(np.mean(image[:, 0, 0]))
            image[..., i] = np.pad(image[..., i],
                                   ((pad[0][0], pad[0][1]), (pad[1][0], pad[1][1])),
                                   constant_values=(pad_value, pad_value),
                                   mode='constant')

        mask = np.pad(mask, ((pad[0][0], pad[0][1]), (pad[1][0], pad[1][1]), (0, 0)), mode='constant')
        y_min, x_min = random.randint(0, H + 128 - self.input_shape[0]), \
                       random.randint(0, W + 128 - self.input_shape[1])

        if self.mode == 'train':
            new_image, new_mask = self.aug(image[y_min:y_min + 256, x_min:x_min + 256, :].copy(),
                                           mask[y_min:y_min + 256, x_min:x_min + 256, :].copy())
        else:
            new_image, new_mask = image[y_min:y_min + 256, x_min:x_min + 256, :], mask[y_min:y_min + 256,
                                                                                  x_min:x_min + 256, :]
        new_image, new_mask = torch.tensor(new_image).permute((2, 0, 1)), torch.tensor(new_mask).permute((2, 0, 1))

        return new_image.float() / 255, new_mask

    def aug(self, image, mask):
        mask = mask.astype(np.uint8)
        mask = SegmentationMapsOnImage(mask, shape=mask.shape)
        seq1 = iaa.Sequential(
            [iaa.Affine(rotate=(-45, 45), shear=5, order=0, scale=(0.8, 1.2), translate_percent=(0.01, 0.01)),
             iaa.Fliplr(), iaa.Flipud()], random_order=True)
        seq2 = iaa.SomeOf((2, 4), [iaa.GaussianBlur(), iaa.MedianBlur(),
                                   iaa.AddToHueAndSaturation(value_hue=[-11, 11], value_saturation=[-10, 10],
                                                             from_colorspace='BGR'), iaa.GammaContrast()])
        seq3 = iaa.Sequential([iaa.ElasticTransformation(alpha=50, sigma=8)])

        seq1_det = seq1.to_deterministic()
        seq2_det = seq2.to_deterministic()
        seq3_det = seq3.to_deterministic()

        images_aug = seq1_det.augment_image(image=image)
        segmaps_aug = seq1_det.augment_segmentation_maps([mask])[0]

        images_aug = seq2_det.augment_image(image=images_aug)
        segmaps_aug = seq2_det.augment_segmentation_maps([segmaps_aug])[0]

        if random.random() > 0.9:
            images_aug = seq3_det.augment_image(image=images_aug)
            segmaps_aug = seq3_det.augment_segmentation_maps([segmaps_aug])[0]

        segmaps_aug = segmaps_aug.get_arr_int().astype(np.uint8)

        return images_aug, segmaps_aug
