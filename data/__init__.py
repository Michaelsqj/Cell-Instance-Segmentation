from typing import List, Dict, Tuple
import torch
from torch.utils.data import Dataset, DataLoader
from .HVDataset import HVDataset

dataset_zoo = {'hv': HVDataset}


def build_dataloader(cfg, mode) -> DataLoader:
    dataset = dataset_zoo[cfg.DATASET.Name](images=cfg.DATASET.Image,
                                            labels=cfg.DATASET.Label if mode == 'train' else None,
                                            input_path=cfg.DATASET.InputPath,
                                            mode=mode,
                                            input_shape=cfg.MODEL.InputShape,
                                            output_shape=cfg.MODEL.OutputShape
                                            )
    dataloader = DataLoader(dataset, batch_size=cfg.SOLVER.BatchSize, shuffle=True, collate_fn=collate_fn)
    return dataloader


def collate_fn(batch):
    images, targets, names = zip(*batch)
    images = torch.stack(images, dim=0)
    targets = torch.stack(targets, dim=0)
    return images, targets, names
