from typing import List, Dict, Tuple
import torch
from torch.utils.data import Dataset, DataLoader
from .HVDataset import HVDataset

dataset_zoo = {'hv': HVDataset}


def build_dataloader(cfg, mode) -> DataLoader:
    dataset = dataset_zoo[cfg.DATASET.NAME](images=cfg.DATASET.IMAGE,
                                            labels=cfg.DATASET.Label if mode == 'train' else None,
                                            input_path=cfg.DATASET.INPUTPATH,
                                            mode=mode,
                                            input_shape=cfg.MODEL.INPUT_SHAPE,
                                            output_shape=cfg.MODEL.OUTPUT_SHAPE,
                                            reject_size=cfg.MODEL.REJECT_SIZE,
                                            reject_p=cfg.MODEL.REJECT_P,
                                            stride=cfg.INFERENCE.STRIDE
                                            )
    collate_fn = collate_fn_train if mode == 'train' else collate_fn_test
    dataloader = DataLoader(dataset, batch_size=cfg.SOLVER.BATCHSIZE, shuffle=True, collate_fn=collate_fn)
    return dataloader


def collate_fn_train(batch):
    images, targets, names = zip(*batch)
    images = torch.stack(images, dim=0)
    targets = torch.stack(targets, dim=0)
    return images, targets, names


def collate_fn_test(batch):
    images, pos, names = zip(*batch)
    images = torch.stack(images, dim=0)
    return images, pos, names
