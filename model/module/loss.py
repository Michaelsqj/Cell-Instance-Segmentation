# PyTorch
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

"""
WeightedBCE : inputs after sigmoid, target have same size as inputs, each channel {0,1}
                can have weight on every channel
CrossEntropy : raw inputs do not convert to 0~1, contains softmax, target do not have the 'C' channel
                CrossEntropy can have channel weight, same size as channel numbers
DiceLoss :  inputs after sigmoid, target have same size as inputs
                can have channel weight
MSELoss : inputs can use sigmoid or not, based on the performance. targets same size as inputs
                no weight
Generalized DiceLoss : #todo
"""
class WeightedBCE(nn.Module):
    """Weighted binary cross-entropy.
    """

    def __init__(self, size_average=True, reduce=True):
        super().__init__()
        self.size_average = size_average
        self.reduce = reduce

    def forward(self, pred, target, weight=None):
        # _assert_no_grad(target)
        return F.binary_cross_entropy(pred, target, weight)


class FocalLoss(nn.Module):
    def __init__(self):
        super(FocalLoss, self).__init__()

    def forward(inputs, targets):
        new_inputs = F.softmax(inputs)
        C = inputs.shape[1]
        tmp = torch.zeros(size=inputs.shape).cuda()
        new_targets = tmp.scatter_(dim=1, index=targets.unsqueeze(dim=1).long(), value=1).cuda()
        total_loss = 0
        for c in range(C):
            total_loss += ((-0.25 * torch.pow((1 - new_inputs[:, c, :, :]), 2) * torch.log(
                new_inputs[:, c, :, c])) * new_targets
                           - 0.75 * torch.pow(new_inputs[:, c, :, :], 2) * torch.log(
                        1 - new_inputs[:, c, :, :]) * (1 - new_targets)).mean()
        return total_loss / C


class CE(nn.Module):
    def __init__(self):
        super(CE, self).__init__()

    def forward(self, input):
        C = input.shape[1]
        total_loss = 0
        for c in range(C):
            total_loss += F.binary_cross_entropy


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor, smooth: float = 1, weight=None) -> torch.Tensor:
        """
        :param inputs:  NxCX...
        :param targets: NXCX... same size as inputs
        :param weight: CX1 tensor
        """
        assert inputs.shape == targets.shape
        N = targets.size(0)
        if weight is not None:
            assert weight.shape[0] == inputs.shape[1]
            loss = 0
            for i in range(inputs.shape[1]):
                input_flat = inputs[:, i].view(N, -1)
                target_flat = targets[:, i].view(N, -1)
                intersection = input_flat * target_flat
                loss += weight[i] * (2 * intersection.sum() + smooth) / (
                        (input_flat * input_flat).sum() + (target_flat * target_flat).sum() + smooth)
            loss = 1 - loss / torch.sum(weight)
        else:
            input_flat = inputs.view(N, -1)
            target_flat = targets.view(N, -1)
            intersection = input_flat * target_flat
            loss = 1 - (2 * intersection.sum() + smooth) / (
                    (input_flat * input_flat).sum() + (target_flat * target_flat).sum() + smooth)
        return loss
