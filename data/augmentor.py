import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
import random
import numpy as np


def aug(image, mask):
    mask = mask.astype(np.uint8)
    mask = SegmentationMapsOnImage(mask, shape=mask.shape)
    seq1 = iaa.Sequential(
        [iaa.Affine(rotate=(-45, 45), shear=5, order=0, scale=(0.8, 1.2), translate_percent=(0.01, 0.01)),
         iaa.Fliplr(), iaa.Flipud()], random_order=True)
    seq2 = iaa.SomeOf((2, 4), [iaa.GaussianBlur(), iaa.MedianBlur(),
                               iaa.AddToHueAndSaturation(value_hue=[-11, 11], value_saturation=[-10, 10],
                                                         from_colorspace='BGR'), iaa.GammaContrast()])
    seq3 = iaa.Sequential([iaa.ElasticTransformation(alpha=50, sigma=8)])

    seq1_det = seq1.to_deterministic()
    seq2_det = seq2.to_deterministic()
    seq3_det = seq3.to_deterministic()

    images_aug = seq1_det.augment_image(image=image)
    segmaps_aug = seq1_det.augment_segmentation_maps([mask])[0]

    images_aug = seq2_det.augment_image(image=images_aug)
    segmaps_aug = seq2_det.augment_segmentation_maps([segmaps_aug])[0]

    if random.random() > 0.9:
        images_aug = seq3_det.augment_image(image=images_aug)
        segmaps_aug = seq3_det.augment_segmentation_maps([segmaps_aug])[0]

    segmaps_aug = segmaps_aug.get_arr_int().astype(np.uint8)

    return images_aug, segmaps_aug
