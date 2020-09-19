import numpy as np
import cv2
from scipy.ndimage.filters import convolve


#################
# get target ####
#################
def get_hv_target(target: np.array) -> np.ndarray:
    """
    Parameters
    ----------
    targets H*W with the value of {0,...,n} n is the number of cells
    Returns 2*H*W
    -------
    """
    assert np.ndim(target) == 2
    hv_target = np.zeros(shape=(2, target.shape[0], target.shape[1]), dtype=float)
    inst_centroid_list, inst_id_list = get_inst_centroid(target)  # [(x1,y1),(x2,y2),(x3,y3)....(xn,yn)]
    for _ in range(len(inst_id_list)):  # id: instance index from 1~n
        xc, yc = inst_centroid_list[_]
        id = inst_id_list[_]
        H, V = np.meshgrid(np.arange(target.shape[1]), np.arange(target.shape[0]))
        xc, yc = int(xc), int(yc)
        tmp_h = H - xc
        tmp_v = V - yc
        tmp_h = np.where(target == id, tmp_h, 0)
        tmp_v = np.where(target == id, tmp_v, 0)
        # rescale to -1~1
        # horizontal
        maximum = np.max(tmp_h)
        minimum = np.min(tmp_h)
        if maximum > 0 and minimum < 0:
            tmp_h_pos = np.where(tmp_h > 0, tmp_h, 0).astype(float)
            tmp_h_neg = np.where(tmp_h < 0, tmp_h, 0).astype(float)
            tmp_h_pos = tmp_h_pos / maximum
            tmp_h_neg = tmp_h_neg / abs(minimum)
            tmp_h = tmp_h_neg + tmp_h_pos
        elif maximum > 0 and minimum == 0:
            tmp_h_pos = np.where(tmp_h > 0, tmp_h, 0).astype(float)
            tmp_h_pos = tmp_h_pos / maximum
            tmp_h = tmp_h_pos.astype(float)
        elif maximum == 0 and minimum < 0:
            tmp_h_neg = np.where(tmp_h < 0, tmp_h, 0).astype(float)
            tmp_h_neg = tmp_h_neg / abs(minimum)
            tmp_h = tmp_h_neg.astype(float)
        else:
            tmp_h = tmp_h.astype(float)
        # vertical
        maximum = np.max(tmp_v)
        minimum = np.min(tmp_v)
        if maximum > 0 and minimum < 0:
            tmp_v_pos = np.where(tmp_v > 0, tmp_v, 0).astype(float)
            tmp_v_neg = np.where(tmp_v < 0, tmp_v, 0).astype(float)
            tmp_v_pos = tmp_v_pos / maximum
            tmp_v_neg = tmp_v_neg / abs(minimum)
            tmp_v = tmp_v_neg + tmp_v_pos
        elif maximum > 0 and minimum == 0:
            tmp_v_pos = np.where(tmp_v > 0, tmp_v, 0).astype(float)
            tmp_v_pos = tmp_v_pos / maximum
            tmp_v = tmp_v_pos
        elif maximum == 0 and minimum < 0:
            tmp_v_neg = np.where(tmp_v < 0, tmp_v, 0).astype(float)
            tmp_v_neg = tmp_v_neg / abs(minimum)
            tmp_v = tmp_v_neg
        else:
            tmp_v = tmp_v.astype(float)

        Temp = np.where(target == id, tmp_h, 0).squeeze()
        tmp = np.where(hv_target[0, :, :] != 0, 1, 0) * np.where(Temp != 0, 1, 0)
        tmp = 1 - tmp
        hv_target[0, :, :] = hv_target[0, :, :] * tmp + Temp

        Temp = np.where(target == id, tmp_v, 0).squeeze()
        tmp = np.where(hv_target[1, :, :] != 0, 1, 0) * np.where(Temp != 0, 1, 0)
        tmp = 1 - tmp
        hv_target[1, :, :] = hv_target[1, :, :] * tmp + Temp

    return hv_target


def get_inst_centroid(inst_map):
    inst_centroid_list = []
    id_list = []
    inst_id_list = np.unique(inst_map)

    for inst_id in inst_id_list:  # avoid 0 i.e background
        if inst_id > 0:
            mask = np.where(inst_map == inst_id, 1, 0)
            inst_moment = cv2.moments(mask.astype(np.int16))
            inst_centroid = ((inst_moment["m10"] / inst_moment["m00"]),  # 横向
                             (inst_moment["m01"] / inst_moment["m00"]))  # 纵向
            inst_centroid_list.append(inst_centroid)
            id_list.append(inst_id)
            # tmp = np.zeros(inst_map.shape)
            # tmp[int(inst_centroid[1]), int(inst_centroid[0])] = 255
            # cv2.imshow(f'{inst_id}', (mask * 122 + tmp).astype(np.uint8))
            # cv2.waitKey(0)
            # cv2.destroyWindow(f'{inst_id}')
    return inst_centroid_list, id_list


def one_hot(label, C):
    target = np.zeros((C, label.shape[0], label.shape[1]), dtype=label.dtype)
    for i in range(C):
        target[i] = (label == i).astype(target.dtype)
    target[C-1] += (label > (C-1)).astype(target.dtype)
    return target


#################
# get weight ####
#################
def weight_binary_ratio(labels, alpha=1.0):
    """Binary-class rebalancing."""
    # labels CxHxW
    weights = []
    for c in range(labels.shape[0]):
        label = labels[c]
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
        weights.append(weight)
    weights = np.stack(weights, axis=0)
    return weights


def weight_gdl(label):
    """
    weight for generalized dice loss https://arxiv.org/pdf/1707.03237.pdf
    label should be C x H x W
    return C x 1 x 1
    """
    weight = np.zeros((label.shape[0], 1, 1))
    for i in range(label.shape[0]):
        w = np.sum(label[i])
        w = np.clip(w, a_min=1, a_max=10)
        w = w * w
        weight[i] = 1 / w
    return weight
