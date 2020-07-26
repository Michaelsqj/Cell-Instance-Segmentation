import numpy as np
import imageio


def read_img(path):
    suf = path.split('.')[-1]
    if suf == 'npy':
        image = np.load(path, allow_pickle=True)
    else:
        image = imageio.imread(path)
    return image
