from torch.utils.data import DataLoader
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
    target = [None] * len(targets[0])
    for i in range(len(targets[0])):
        temp = []
        for n in range(len(targets)):
            temp.append(targets[n][i])
        target[i] = np.stack(temp, axis=0)
    weight = [None] * len(weights)
    for i in range(len(weights[0])):
        weight[i] = [None] * len(weights[0])
        for j in range(len(weights[0][i])):
            temp = []
            for n in range(len(weights)):
                temp.append(weights[n][i][j])
            weight[i][j] = np.stack(temp, axis=0)
    return images, target, weight


def collate_fn_test(batch):
    images, pos = zip(*batch)
    images = np.stack(images, axis=0)
    return images, pos
