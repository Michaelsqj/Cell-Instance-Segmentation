import os
from torch.utils.data import Dataset
import random
import math
import json
from Segmentation.utils import read_img
from Segmentation.data.utils import *
from Segmentation.data.augmentor import augmentor


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
        dataset = cfg.DATASET.INPUT_PATH + cfg.DATASET.IMAGE if mode == 'train' else cfg.DATASET.INPUT_PATH + cfg.INFERENCE.IMAGE
        self.images = json.load(open(dataset, 'r'))['image']
        self.labels = json.load(open(dataset, 'r'))['label'] if mode == 'train' else None
        self.input_path = cfg.DATASET.INPUT_PATH
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
                image = read_img(self.input_path + self.images[idx])
                self.pos.extend(self.compute_pos(image.shape, idx))
                self.shapes.append(image.shape[:2])

    def __getitem__(self, idx: int):
        if self.mode == 'train':
            """
            read image -> pad image -> sample image -> aug 
            """
            image = read_img(os.path.join(self.input_path, self.images[idx]))
            mask = read_img(os.path.join(self.input_path, self.labels[idx]))
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
            image = self.sample(pos[1:], image)
            return image, pos

    def __len__(self):
        if self.mode == 'train':
            return len(self.images)
        if self.mode == 'test':
            return len(self.pos)

    def pad(self, image, pad):
        a = pad[0]
        b = pad[1]
        temp = []
        for i in range(image.shape[2]):
            image0 = np.squeeze(image[..., i])
            v = int(np.mean(image[:, 0, i]))
            image0 = np.pad(image0, ((a[0], b[0]), (a[1], b[1])), constant_values=(v, v), mode='constant')
            temp.append(np.expand_dims(image0, axis=2))
        image = np.concatenate(temp, axis=2)
        return image

    def compute_pos(self, image_shape, idx=None):
        if self.mode == 'test':
            # only pad at the end
            pos = [[], []]
            for i in range(2):
                if image_shape[i] < self.output_shape[i]:
                    N = 0
                else:
                    N = math.ceil((image_shape[i] - self.output_shape[i]) / float(self.stride))
                for j in range(N):
                    pos[i].append(j * self.stride)
            if idx is None:
                pos1 = []
                for i in range(len(pos[0])):
                    for j in range(len(pos[1])):
                        pos1.append((pos[0][i], pos[1][j]))
                return pos1
            else:
                pos1 = []
                for i in range(len(pos[0])):
                    for j in range(len(pos[1])):
                        pos1.append((idx, pos[0][i], pos[1][j]))
                return pos1
        if self.mode == 'train':
            pos = [None] * 2
            for i in range(2):
                if image_shape[i] < self.output_shape[i]:
                    pos[i] = 0
                else:
                    pos[i] = random.randint(0, self.output_shape[i])
            return pos

    def sample(self, pos, image, mask=None):
        pad_image = [[0, 0], [0, 0]]
        if mask is not None:
            pad_mask = [[0, 0], [0, 0]]
        pos1 = [[0, 0], [0, 0]]
        for i in range(2):
            pos1[i][0] = max(0, pos[i] - (self.input_shape[i] - self.output_shape[i]) // 2)
            pad_image[i][0] = max(0, (self.input_shape[i] - self.output_shape[i]) // 2 - pos[i])
            pos1[i][1] = min(image.shape[i], pos[i] + (self.input_shape[i] + self.output_shape[i]) // 2)
            pad_image[i][1] = max(0, pos[i] + (self.input_shape[i] + self.output_shape[i]) // 2 - image.shape[i])
            # image < output shape
            if mask is not None:
                pad_mask[i][1] = max(0, pos[i] + self.output_shape[i] - image.shape[i])
        image = image[pos[0][0]:pos[0][1], pos[1][0]:pos[1][1], :]
        image = self.pad(image, pad_image)
        if mask is not None:
            mask = mask[pos[0][0]:pos[0][1], pos[1][0]:pos[1][1], :]
            mask = self.pad(mask, pad_mask)
            return image, mask
        else:
            return image
