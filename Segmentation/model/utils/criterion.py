import torch
import torch.nn as nn
from Segmentation.model.module.loss import DiceLoss, WeightedBCE, GradMSE, GDLLoss


class Criterion(nn.Module):
    """
    compute loss given by loss option and loss weight
    most important : each input should correspond to each target
    1 input -> 1 target -> >=1 loss
    """

    def __init__(self, lopt, wopt, loss_weigth, device):
        """
        :param lopt: 'str' choose the loss function to use
        :param wopt: list of float choose loss weight
        :param copt: list of float choose weight of different channels
        """
        super(Criterion, self).__init__()
        self.loss = self.get_loss(lopt)
        self.total_loss = 0
        self.losses = []  # the number of each loss, for monitor purpose
        self.loss_weight = loss_weigth
        self.wopt = wopt
        self.device = device

    def get_loss(self, lopt):
        loss = [None] * len(lopt)
        for i in range(len(lopt)):
            loss[i] = [None] * len(lopt[i])
            for j in range(len(lopt[i])):
                if lopt[i][j] == 'CELoss':
                    loss[i][j] = nn.CrossEntropyLoss()
                elif lopt[i][j] == 'DiceLoss':
                    loss[i][j] = DiceLoss()
                elif lopt[i][j] == 'MSELoss':
                    loss[i][j] = nn.MSELoss()
                elif lopt[i][j] == 'WeightedBCE':
                    loss[i][j] = WeightedBCE()
                elif lopt[i][j] == 'GradMSE':
                    loss[i][j] = GradMSE()
                elif lopt[i][j] == 'GDL':
                    loss[i][j] = GDLLoss()
                else:
                    raise ValueError('loss option not exist')
        return loss

    def forward(self, inputs, targets, weights):
        """
        :param inputs:  [ ] * n
        :param targets:  [ ] * n
        :return:
        """
        for i in range(len(self.loss)):
            for j in range(len(self.loss[i])):
                if self.wopt[i][j] == '0':  # no loss weight
                    loss = self.loss_weight[i][j] * self.loss[i][j](inputs,
                                                                    torch.from_numpy(targets[i]).to(self.device))
                else:
                    loss = self.loss_weight[i][j] * self.loss[i][j](inputs,
                                                                    torch.from_numpy(targets[i]).to(self.device),
                                                                    torch.from_numpy(weights[i][j]).to(self.device))
                self.total_loss += loss
                self.losses.append(loss.detach().cpu().item())
        return self.total_loss, self.losses
