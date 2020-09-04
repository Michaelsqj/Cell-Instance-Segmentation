import torch
import json
import os
from tqdm import tqdm
import numpy as np
from ..model import build_model, Criterion, build_solver, visualizer
from ..data import build_dataloader
from ..utils import save_img
import logging

logger = logging.getLogger('main.trainer')


class Trainer():
    def __init__(self, cfg, mode, device, checkpoint):
        self.device = device
        self.mode = mode
        self.cfg = cfg
        self.criterion = Criterion(self.cfg.MODEL.lopt, self.cfg.MODEL.wopt, self.cfg.MODEL.copt)
        self.model = build_model(self.cfg.MODEL.NAME)
        self.dataloader = build_dataloader(self.cfg, self.mode)
        self.optimizer, self.lr_scheduler = build_solver(cfg, self.model)
        self.start_iter = 0
        self.vis = visualizer(self.cfg.DATASET.OUTPUT_PATH)
        if checkpoint is not None:
            self.update_checkpoint(checkpoint)

    def train(self):
        for iter in range(self.start_iter, self.cfg.SOLVER.ITERATION_TOTAL):
            for image, mask in tqdm(self.dataloader):
                image, mask = image.to(self.device), mask.to(self.device)
                pred = self.model(image)
                loss, losses = self.criterion(pred, mask)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
            if (iter + 1) % self.cfg.SOLVER.ITERATION_SAVE == 0:
                self.save_checkpoint(iter)
                print('save model', iter + 1)
            # update learning rate
            self.lr_scheduler.step(
                loss) if self.cfg.SOLVER.LR_SCHEDULER == 'ReduceLROnPlateau' else self.lr_scheduler.step(iter)
            if (iter + 1) % self.cfg.MODEL.VIS_ITER == 0:
                self.vis.vis(iter, losses, image.detach().cpu(), pred.detach().cpu(), mask.detach().cpu())

    def test(self):
        filenames = json.load(self.cfg.DATASET.INPUTPATH + self.cfg.INFERENCE.IMAGE)['image']
        current = ''  # record current image name
        result = None
        weight = None
        self.model.eval()
        for image, pos in tqdm(self.dataloader):
            image = image.to(self.device)
            name = filenames[pos[0]]
            pred = self.model(image).detach().cpu().numpy()
            if name is not current:
                if result is not None and weight is not None:
                    result = result / weight
                    save_img(self.cfg.INFERENCE.OUTPUT_PATH + current, result)
                    print('save result', current)
                    current = name
                    print('start predict result', current)
                shape = self.dataloader.datset.shapes[pos[0]]
                result = np.zeros([image.shape[1], shape[0], shape[1]])
                for i in range(image.shape[0]):
                    result[:, pos[0]:min(shape[0], pos[0] + self.cfg.MODEL.OUTPUT_SHAPE[0]),
                    pos[1]:min(shape[1], pos[1] + self.cfg.MODEL.OUTPUT_SHAPE[1])] += pred
                    weight[:, pos[0]:min(shape[0], pos[0] + self.cfg.MODEL.OUTPUT_SHAPE[0]),
                    pos[1]:min(shape[1], pos[1] + self.cfg.MODEL.OUTPUT_SHAPE[1])] += 1
            else:
                for i in range(image.shape[0]):
                    result[:, pos[0]:min(shape[0], pos[0] + self.cfg.MODEL.OUTPUT_SHAPE[0]),
                    pos[1]:min(shape[1], pos[1] + self.cfg.MODEL.OUTPUT_SHAPE[1])] += pred
                    weight[:, pos[0]:min(shape[0], pos[0] + self.cfg.MODEL.OUTPUT_SHAPE[0]),
                    pos[1]:min(shape[1], pos[1] + self.cfg.MODEL.OUTPUT_SHAPE[1])] += 1
        print('prediction complete')

    def save_checkpoint(self, iteration):
        state = {'iteration': iteration + 1,
                 'state_dict': self.model.module.state_dict(),  # Saving torch.nn.DataParallel Models
                 'optimizer': self.optimizer.state_dict(),
                 'lr_scheduler': self.lr_scheduler.state_dict()}
        # Saves checkpoint to experiment directory
        filename = 'checkpoint_%04d.pth.tar' % (iteration + 1)
        filename = os.path.join(self.cfg.DATASET.OUTPUT_PATH, filename)
        torch.save(state, filename)

    def update_checkpoint(self, checkpoint):
        # load pre-trained model
        print('Load pretrained checkpoint: ', checkpoint)
        checkpoint = torch.load(checkpoint)
        print('checkpoints: ', checkpoint.keys())

        # update model weights
        if 'state_dict' in checkpoint.keys():
            pretrained_dict = checkpoint['state_dict']
            model_dict = self.model.module.state_dict()  # nn.DataParallel
            # 1. filter out unnecessary keys
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            # 2. overwrite entries in the existing state dict
            model_dict.update(pretrained_dict)
            # 3. load the new state dict
            self.model.module.load_state_dict(model_dict)  # nn.DataParallel

        if not self.cfg.SOLVER.ITERATION_RESTART:
            # update optimizer
            if 'optimizer' in checkpoint.keys():
                self.optimizer.load_state_dict(checkpoint['optimizer'])

            # update lr scheduler
            if 'lr_scheduler' in checkpoint.keys():
                self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

            # load iteration
            if 'iteration' in checkpoint.keys():
                self.start_iter = checkpoint['iteration']
