from typing import Tuple
import torch
from torch.nn import functional as F
import numpy as np
import cv2


def get_gradient_hv(logits: torch.Tensor,
                    h_ch: int = 0,
                    v_ch: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
    """Get horizontal & vertical gradients

    Parameters
    ----------
    logits : torch.Tensor
        Raw logits from HV branch
    h_ch : int
        Number of horizontal channels
    v_ch : int
        Number of vertical channels

    Returns
    -------
    gradients : Tuple[torch.Tensor, torch.Tensor]
        Horizontal and vertical gradients
    """
    mh = torch.tensor([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]], dtype=torch.float).unsqueeze(dim=0).unsqueeze(dim=1).cuda()
    mv = torch.tensor([[1, 2, 1],
                       [0, 0, 0],
                       [-1, -2, -1]], dtype=torch.float).unsqueeze(dim=0).unsqueeze(dim=1).cuda()

    hl = logits[:, h_ch, :, :].unsqueeze(dim=1).float()
    vl = logits[:, v_ch, :, :].unsqueeze(dim=1).float()

    assert (mh.dim() == 4 and mv.dim() == 4 and hl.dim() == 4 and vl.dim() == 4)

    dh = F.conv2d(hl, mh, stride=1, padding=1)
    dv = F.conv2d(vl, mv, stride=1, padding=1)

    return dh, dv


def get_np_targets(targets: torch.Tensor) -> torch.Tensor:
    '''

    Parameters
    ----------
    targets N*H*W with the value of {0,...,n} n is the number of cells

    Returns N*H*W with 0:background 1:cells
    -------

    '''
    assert targets.dim() == 3
    # targets = targets.double()
    # np_targets = targets.where(targets == 0, torch.tensor(1).double())
    np_targets = np.zeros(targets.shape)
    N = targets.shape[0]
    for j in range(N):
        inst_map = (targets[j, ...]).numpy()
        inst_id = list(np.unique(inst_map))
        inst_id_list = []
        for i in inst_id:
            if i % 1.0 == 0 and i > 0:
                inst_id_list.append(i)
        id_list = []
        for inst_id in inst_id_list[0:]:  # avoid 0 i.e background
            mask = np.array(inst_map == inst_id, np.int16)
            mask = np.squeeze(mask)
            tmp = mask.copy()
            _, tmp = cv2.connectedComponents(tmp.astype(np.uint8))
            max_s = 0
            for k in np.unique(tmp):
                if k > 0:
                    temp = np.where(tmp == k, 1, 0)
                    if np.sum(temp) > max_s:
                        max_s = np.sum(temp)
                        mask = temp * inst_id
            if max_s < 20:
                continue
            np_targets[j, ...] += np.where(mask > 0, 1, 0)
    np_targets = torch.tensor(np_targets)
    return np_targets


def get_hv_targets(target: np.array) -> np.ndarray:
    """
    Parameters
    ----------
    targets H*W with the value of {0,...,n} n is the number of cells
    Returns 2*H*W
    -------
    """
    assert np.ndim(target) == 2
    hv_targets = np.zeros(shape=(2, target.shape[0], target.shape[1]), dtype=float)
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
        #### rescale to -1~1
        #### horizontal
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
        #### vertical
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
        tmp = np.where(hv_targets[0, :, :] != 0, 1, 0) * np.where(Temp != 0, 1, 0)
        tmp = 1 - tmp
        hv_targets[0, :, :] = hv_targets[0, :, :] * tmp + Temp

        Temp = np.where(target == id, tmp_v, 0).squeeze()
        tmp = np.where(hv_targets[1, :, :] != 0, 1, 0) * np.where(Temp != 0, 1, 0)
        tmp = 1 - tmp
        hv_targets[1, :, :] = hv_targets[1, :, :] * tmp + Temp

    # assert hv_targets.dim() == 4 and hv_targets.shape[1] == 2
    return hv_targets


def get_nc_targets(targets: torch.Tensor) -> torch.Tensor:
    return torch.tensor()


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


def weight_gdl(label):
    """
    weight for generalized dice loss https://arxiv.org/pdf/1707.03237.pdf
    label should be C x ...
    """
    weight = np.zeros((label.shape[0], 1))
    for i in range(label.shape[0]):
        w = np.sum(label[i])
        w = np.clip(w, a_min=1, a_max=10)
        w = w * w
        weight[i] = w
    return weight
