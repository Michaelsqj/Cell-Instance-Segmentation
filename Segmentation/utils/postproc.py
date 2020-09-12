import cv2
import numpy as np
from scipy.ndimage import filters, measurements
from scipy.ndimage.morphology import (binary_dilation, binary_fill_holes,
                                      distance_transform_cdt,
                                      distance_transform_edt)
from skimage.morphology import remove_small_objects, watershed


def proc_np_hv(pred, marker_mode=2, energy_mode=2, rgb=None):
    """
    Process Nuclei Prediction with XY Coordinate Map
    Args:
        pred: prediction output, assuming
                channel 0 contain probability map of nuclei
                channel 1 containing the regressed X-map
                channel 2 containing the regressed Y-map
    """
    assert marker_mode == 2 or marker_mode == 1, 'Only support 1 or 2'
    assert energy_mode == 2 or energy_mode == 1, 'Only support 1 or 2'

    blb_raw = pred[..., 0]
    h_dir_raw = pred[..., 1]
    v_dir_raw = pred[..., 2]

    ##### Processing
    blb = np.copy(blb_raw)
    blb[blb >= 0.5] = 1
    blb[blb < 0.5] = 0

    blb = measurements.label(blb)[0]
    blb = remove_small_objects(blb, min_size=10)
    blb[blb > 0] = 1  # back ground is 0 already
    #####

    if energy_mode == 2 or marker_mode == 2:
        h_dir = cv2.normalize(h_dir_raw, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        v_dir = cv2.normalize(v_dir_raw, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        sobelh = cv2.Sobel(h_dir, cv2.CV_64F, 1, 0, ksize=21)
        sobelv = cv2.Sobel(v_dir, cv2.CV_64F, 0, 1, ksize=21)

        sobelh = 1 - (cv2.normalize(sobelh, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F))
        sobelv = 1 - (cv2.normalize(sobelv, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F))

        overall = np.maximum(sobelh, sobelv)
        overall = overall - (1 - blb)
        overall[overall < 0] = 0

        if energy_mode == 2:
            dist = (1.0 - overall) * blb
            ## nuclei values form mountains so inverse to get basins
            dist = -cv2.GaussianBlur(dist, (3, 3), 0)

        if marker_mode == 2:
            overall[overall >= 0.4] = 1
            overall[overall < 0.4] = 0

            marker = blb - overall
            marker[marker < 0] = 0
            marker = binary_fill_holes(marker).astype('uint8')
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            marker = cv2.morphologyEx(marker, cv2.MORPH_OPEN, kernel)
            marker = measurements.label(marker)[0]
            marker = remove_small_objects(marker, min_size=10)

    if energy_mode == 1:
        dist = h_dir_raw * h_dir_raw + v_dir_raw * v_dir_raw
        dist[blb == 0] = np.amax(dist)
        # nuclei values are already basins
        dist = filters.maximum_filter(dist, 7)
        dist = cv2.GaussianBlur(dist, (3, 3), 0)

    if marker_mode == 1:
        h_marker = np.copy(h_dir_raw)
        v_marker = np.copy(v_dir_raw)
        h_marker = np.logical_and(h_marker < 0.075, h_marker > -0.075)
        v_marker = np.logical_and(v_marker < 0.075, v_marker > -0.075)
        marker = np.logical_and(h_marker > 0, v_marker > 0) * blb
        marker = binary_dilation(marker, iterations=2)
        marker = binary_fill_holes(marker)
        marker = measurements.label(marker)[0]
        marker = remove_small_objects(marker, min_size=10)

    proced_pred = watershed(dist, marker, mask=blb)
    pred_inst = remap_label(proced_pred, by_size=True)
    return pred_inst


def pred_class(pred_type, pred_inst):
    """
    :param pred_type: class prediction for each pixel
    :param pred_inst: instance map after watershed and remap
    :return:
    """
    pred_id_list = list(np.unique(pred_inst))[1:]  # exclude background ID
    pred_inst_type = np.full(len(pred_id_list), 0, dtype=np.int32)
    for idx, inst_id in enumerate(pred_id_list):
        inst_type = pred_type[pred_inst == inst_id]
        type_list, type_pixels = np.unique(inst_type, return_counts=True)
        type_list = list(zip(type_list, type_pixels))
        type_list = sorted(type_list, key=lambda x: x[1], reverse=True)
        inst_type = type_list[0][0]
        if inst_type == 0:  # ! pick the 2nd most dominant if exist
            if len(type_list) > 1:
                inst_type = type_list[1][0]
            else:
                print('[Warn] Instance has `background` type')
        pred_inst_type[idx] = inst_type
    return pred_inst_type


def remap_label(pred, by_size=False):
    """
    Rename all instance id so that the id is contiguous i.e [0, 1, 2, 3]
    not [0, 2, 4, 6]. The ordering of instances (which one comes first)
    is preserved unless by_size=True, then the instances will be reordered
    so that bigger nucler has smaller ID
    Args:
        pred    : the 2d array contain instances where each instances is marked
                  by non-zero integer
        by_size : renaming with larger nuclei has smaller id (on-top)
    """
    pred_id = list(np.unique(pred))
    pred_id.remove(0)
    if len(pred_id) == 0:
        return pred  # no label
    if by_size:
        pred_size = []
        for inst_id in pred_id:
            size = (pred == inst_id).sum()
            pred_size.append(size)
        # sort the id by size in descending order
        pair_list = zip(pred_id, pred_size)
        pair_list = sorted(pair_list, key=lambda x: x[1], reverse=True)
        pred_id, pred_size = zip(*pair_list)

    new_pred = np.zeros(pred.shape, np.int32)
    for idx, inst_id in enumerate(pred_id):
        new_pred[pred == inst_id] = idx + 1
    return new_pred
