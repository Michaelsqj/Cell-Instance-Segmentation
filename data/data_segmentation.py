import numpy as np


def get_targets(label, topt):
    for t in topt:
        if t == '0':  # binary seg
            pass
        elif t == '1':  # multi-channel binary seg
            pass
        elif t == '2':  # h-v seg
            pass
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