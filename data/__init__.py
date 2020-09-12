from typing import List, Dict, Tuple
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from .HVDataset import HVDataset

dataset_zoo = {'hv': HVDataset}


def build_dataloader(cfg, mode):
    dataset = dataset_zoo[cfg.DATASET.NAME](cfg, mode)
    collate_fn = collate_fn_train if mode == 'train' else collate_fn_test
    shuffle = True if mode == 'train' else False
    dataloader = DataLoader(dataset, batch_size=cfg.SOLVER.BATCHSIZE, shuffle=shuffle, collate_fn=collate_fn)
    return dataloader


def collate_fn_train(batch):
    images, targets, weights = zip(*batch)
    images = np.stack(images, axis=0)
    # multiple targets for each input
    targets = [np.stack(targets[:][i], axis=0) for i in range(len(targets[0]))]
    weights = [[np.stack(targets[:][i][j], axis=0) for j in range(len(targets[0][i]))] for i in
               range(len(targets[0]))]
    return images, targets, weights


def collate_fn_test(batch):
    images, pos = zip(*batch)
    images = np.stack(images, axis=0)
    return images, pos
