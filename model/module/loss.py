# PyTorch
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np


def weight_binary_ratio(label, alpha=1.0):
    """Binary-class rebalancing."""
    # input: numpy tensor
    # weight for smaller class is 1, the bigger one is at most 100*alpha
    if label.max() == label.min():  # uniform weights for volume with a single label
        weight_factor = 1.0
        weight = np.ones_like(label, np.float32)
    else:
        weight_factor = float(label.sum()) / np.prod(label.shape)

        weight_factor = np.clip(weight_factor, a_min=5e-2, a_max=0.99)

        if weight_factor > 0.5:
            weight = label + alpha * weight_factor / (1 - weight_factor) * (1 - label)
        else:
            weight = alpha * (1 - weight_factor) / weight_factor * label + (1 - label)
    return weight


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

    def forward(self, input: torch.Tensor, target: torch.Tensor, smooth: float = 1e-3) -> torch.Tensor:
        assert input.dim() == 3
        assert target.dim() == 3
        N = target.size(0)

        input_flat = input.view(N, -1)
        target_flat = target.view(N, -1)

        intersection = input_flat * target_flat

        loss = 1 - (2 * intersection.sum() + smooth) / (
                (input_flat * input_flat).sum() + (target_flat * target_flat).sum() + smooth)

        return loss


class DiceCoeff(nn.Module):
    def __init__(self):
        super(DiceCoeff, self).__init__()
        self.dice = DiceLoss()

    def forward(self,
                inputs: torch.Tensor,
                targets: torch.Tensor,
                smooth: float = 1e-3) -> torch.Tensor:
        assert targets.dim() == 3
        tmp = torch.zeros(size=inputs.shape).cuda()
        targets = tmp.scatter_(dim=1, index=targets.unsqueeze(dim=1).long(), value=1).cuda()
        assert inputs.shape[1] in (2, 5)
        assert targets.shape == inputs.shape
        totalloss = 0
        for c in range(inputs.shape[1]):
            totalloss += self.dice(inputs[:, c, :, :].squeeze(), targets[:, c, :, :].squeeze())
        return totalloss.cuda() / (inputs.shape[1])
