import torch
import torch.nn as nn
from model.module.loss import FocalLoss, CE, DiceLoss, WeightedBCE


class Criterion(nn.Module):
    """
    compute loss given by loss option and loss weight
    """

    def __init__(self, lopt, wopt, copt):
        """
        :param lopt: 'str' choose the loss function to use
        :param wopt: list of float choose loss weight
        :param copt: list of float choose weight of different channels
        """
        super(Criterion, self).__init__()
        self.loss = self.get_loss(lopt)
        self.total_loss = 0
        self.losses = []  # the number of each loss, for monitor purpose
        self.weight = wopt
        self.copt = copt

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
                else:
                    raise ValueError('loss option not exist')
        return loss

    def forward(self, inputs, targets):
        """
        :param inputs:  list of model output
        :param targets:  list of target, same number as inputs
        :return:
        """
        for i in range(len(inputs)):
            for j in range(len(self.loss[i])):
                if len(self.copt[i][j]) == 0:
                    temp = self.weight[i] * self.loss[i][j](inputs[i], targets[i])
                else:
                    temp = self.weight[i] * self.loss[i][j](inputs[i], targets[i], self.copt[i][j])
                self.total_loss += temp
                self.losses.append(temp.detach().cpu().item())
        return self.total_loss, self.losses
