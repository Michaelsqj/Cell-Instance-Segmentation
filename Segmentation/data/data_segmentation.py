import numpy as np
from .utils import *


def get_targets(label, topt, copt):
    # label HxWxC
    # return targets [CxHxW,...]
    c={'0':2,'1':5}
    assert np.ndim(label) == 3
    targets = [None] * len(topt)
    for i, t in enumerate(topt):
        if t == '0':  # binary seg
            targets[i] = one_hot(label[..., copt[i]], c[t]).astype(float)
        elif t == '1':  # one_hot
            assert label.shape[-1] > 1
            targets[i] = one_hot(label[..., copt[i]], c[t]).astype(float)
        elif t == '2':  # h-v
            targets[i] = get_hv_target(label[..., copt[i]])
        else:
            raise ValueError('no such target')
    return targets


def get_weights(labels, wopt, mask=None):
    # labels: [CxHxW,...]
    # return : [CxHxW, ...]
    weights = [None] * len(wopt)
    for i in range(len(wopt)):
        weights[i] = [None] * len(wopt[i])
        for j in range(len(wopt[i])):
            if wopt[i][j] == '0':  # no weight
                weights[i][j] = np.ones((1), dtype=float)
            elif wopt[i][j] == '1':  # weight binary
                weights[i][j] = weight_binary_ratio(labels[i])
            elif wopt[i][j] == '2':  # general dice loss
                weights[i][j] = weight_gdl(labels[i])
            elif wopt[i][j] == '3':  # grad hv-> binary
                weights[i][j] = (mask[None, ...] > 0).astype(float)
            else:
                raise ValueError('no such weight')
    return weights
