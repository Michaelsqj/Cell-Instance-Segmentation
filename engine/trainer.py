import torch
import json
from ..model import build_model, Criterion
from ..data import build_dataloader


class Trainer():
    def __init__(self, cfg, mode, device):
        self.mode = mode
        self.cfg = cfg
        self.criterion = Criterion(self.cfg.MODEL.lopt, self.cfg.MODEL.wopt, self.cfg.MODEL.copt)
        self.model = build_model(self.cfg.MODEL.Name)
        self.dataloader = build_dataloader(self.cfg, self.mode)

    def train(self):
        pass

    def test(self):
        pass
