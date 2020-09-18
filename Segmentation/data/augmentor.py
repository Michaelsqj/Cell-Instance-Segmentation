import imgaug.augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
import torch.nn as nn

""" augmentation
https://imgaug.readthedocs.io/en/latest/source/examples_segmentation_maps.html
https://nbviewer.jupyter.org/github/aleju/imgaug-doc/blob/master/notebooks/B05%20-%20Augment%20Segmentation%20Maps.ipynb
non-spatial augmentation : not on mask
    color, noise
spatial augmentation : also on mask
    flip, rotation, resize, translate
image: uint8
mask: uint16
"""


class augmentor(nn.Module):
    def __init__(self, cfg):
        """
        determine from config file
        """
        super(augmentor, self).__init__()
        self.aug_spatial = self.spatial(cfg)
        self.aug_non_spatial = self.non_spatial(cfg)

    def forward(self, data):
        aug1 = self.aug_non_spatial.to_deterministic()
        aug2 = self.aug_spatial.to_deterministic()
        data['image'] = aug1(image=data['image'])
        if 'label' in data.keys() and data['label'] is not None:
            segmap = SegmentationMapsOnImage(data['lable'], shape=data['image'].shape)
            data['image'], segmap = aug2(image=data['image'], segmentation_maps=segmap)
            data['label'] = segmap.get_arr()
        else:
            data['image'] = aug2(image=data['image'])
        return data

    def spatial(self, cfg):
        aug = iaa.Sequential(
            [iaa.Affine(rotate=(-45, 45), shear=5, order=0, scale=(0.8, 1.2), translate_percent=(0.01, 0.01)),
             iaa.Fliplr(),
             iaa.Flipud()],
            random_order=True)
        return aug

    def non_spatial(self, cfg):
        aug = iaa.Sequential(
            [iaa.AddToHueAndSaturation(value_hue=[-11, 11], value_saturation=[-10, 10], from_colorspace='BGR'),
             iaa.GammaContrast()])
        # iaa.GammaContrast((0.5, 2.0), per_channel=True)
        return aug

# def aug(image, mask):
#     mask = mask.astype(np.uint8)
#     mask = SegmentationMapsOnImage(mask, shape=mask.shape)
#     seq1 = iaa.Sequential(
#         [iaa.Affine(rotate=(-45, 45), shear=5, order=0, scale=(0.8, 1.2), translate_percent=(0.01, 0.01)),
#          iaa.Fliplr(), iaa.Flipud()], random_order=True)
#     seq2 = iaa.SomeOf((2, 4), [iaa.GaussianBlur(), iaa.MedianBlur(),
#                                iaa.AddToHueAndSaturation(value_hue=[-11, 11], value_saturation=[-10, 10],
#                                                          from_colorspace='BGR'), iaa.GammaContrast()])
#     seq3 = iaa.Sequential([iaa.ElasticTransformation(alpha=50, sigma=8)])
#
#     seq1_det = seq1.to_deterministic()
#     seq2_det = seq2.to_deterministic()
#     seq3_det = seq3.to_deterministic()
#
#     images_aug = seq1_det.augment_image(image=image)
#     segmaps_aug = seq1_det.augment_segmentation_maps([mask])[0]
#
#     images_aug = seq2_det.augment_image(image=images_aug)
#     segmaps_aug = seq2_det.augment_segmentation_maps([segmaps_aug])[0]
#
#     if random.random() > 0.9:
#         images_aug = seq3_det.augment_image(image=images_aug)
#         segmaps_aug = seq3_det.augment_segmentation_maps([segmaps_aug])[0]
#
#     segmaps_aug = segmaps_aug.get_arr_int().astype(np.uint8)
#
#     return images_aug, segmaps_aug
