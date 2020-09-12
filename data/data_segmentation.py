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
            targets[i] = get_hv_targets(label[..., 0])
        else:
            raise ValueError('no such target')


def get_weight(labels, wopt):
    for i in range(len(wopt)):
        for j in range(len(wopt[i])):
            if wopt[i][j] == '0':  # no weight
                pass
            elif wopt[i][j] == '1':  # weight binary
                pass
            elif wopt[i][j] == '2':  # dice binary
                pass
            else:
                raise ValueError('no such weight')


def one_hot(label):
    C = np.max(label)
    target = np.zeros((C + 1, label.shape[0], label.shape[1]), dtype=label.dtype)
    for i in range(C):
        target[C] = (label == C).astype(target.dtype)
    return target
