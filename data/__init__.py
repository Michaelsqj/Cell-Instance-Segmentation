from typing import List, Dict, Tuple
import torch
from torch.utils.data import Dataset, DataLoader
from .HVDataset import HVDataset

dataset_zoo = {'hv': HVDataset}


def build_dataloader(cfg, mode):
    dataset = dataset_zoo[cfg.DATASET.NAME](cfg, mode)
    collate_fn = collate_fn_train if mode == 'train' else collate_fn_test
    shuffle = True if mode == 'train' else False
    dataloader = DataLoader(dataset, batch_size=cfg.SOLVER.BATCHSIZE, shuffle=shuffle, collate_fn=collate_fn)
    return dataloader


def collate_fn_train(batch):
    images, targets = zip(*batch)
    images = torch.stack(images, dim=0)
    targets = torch.stack(targets, dim=0)
    return images, targets


def collate_fn_test(batch):
    images, pos = zip(*batch)
    images = torch.stack(images, dim=0)
    return images, pos
