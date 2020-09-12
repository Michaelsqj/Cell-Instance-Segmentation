import numpy as np
from .utils import *


def get_targets(label, topt):
    # label HxWx2
    assert np.ndim(label) == 2
    targets = [None] * len(topt)
    for i, t in enumerate(topt):
        if t == '0':  # binary seg
            targets[i] = (label[..., 0] > 0).astype(label.dtype)
        elif t == '1':  # one_hot
            targets[i] = one_hot(label[..., 1])
        elif t == '2':  # h-v seg
            targets[i] = get_hv_target(label[..., 0])
        else:
            raise ValueError('no such target')


def get_weight(labels, wopt):
    # labels: CxHxW  1st channel instance index, *2nd channel class index
    weights = [None] * len(wopt)
    for i in range(len(wopt)):
        weights[i] = [None] * len(wopt[i])
        for j in range(len(wopt[i])):
            if wopt[i][j] == '0':  # no weight
                weights[i][j] = np.ones((1), dtype=float)
            elif wopt[i][j] == '1':  # weight binary
                weights[i][j] = weight_binary_ratio(labels[0])
            elif wopt[i][j] == '2':  # general dice loss
                weights[i][j] = weight_gdl(labels[1])
            elif wopt[i][j] == '3':  # grad hv-> binary
                weights[i][j] = (labels[0] > 0).astype(float)
            else:
                raise ValueError('no such weight')
