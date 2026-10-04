"""Author augmentation pattern, used by brainIAC_pretraining.train."""

import numpy as np
from monai.transforms import (
    Compose, LoadImaged, NormalizeIntensityd, RandAdjustContrastd,
    RandAffined, RandBiasFieldd, RandCoarseDropoutd, RandFlipd,
    RandGaussianNoised, RandGaussianSmoothd, Resized, ToTensord,
)

from .random_resized_crop import RandomResizedCrop3D


def build_transform(size=(96, 96, 96)):
    keys = ["image"]
    return Compose([
        LoadImaged(keys=keys, ensure_channel_first=True),
        Resized(keys=keys, spatial_size=tuple(size)),
        RandBiasFieldd(keys=keys, prob=0.3),
        NormalizeIntensityd(keys=keys, nonzero=True, channel_wise=True),
        RandomResizedCrop3D(keys=keys),
        RandFlipd(keys=keys, spatial_axis=[0], prob=0.5),
        RandAffined(
            keys=keys, rotate_range=(np.pi / 12, np.pi / 12, 0),
            translate_range=((-10, 10), (-10, 10), (-5, 5)),
            scale_range=((0.85, 1.15), (0.85, 1.15), (0.9, 1.1)),
            padding_mode="border", prob=0.7,
        ),
        RandGaussianSmoothd(keys=keys, prob=0.5, sigma_x=(0.5, 1.5),
                            sigma_y=(0.5, 1.5), sigma_z=(0.5, 1.5)),
        RandCoarseDropoutd(keys=keys, prob=0.5, holes=2,
                          spatial_size=(2, 2, 2), max_spatial_size=(4, 4, 4), fill_value=0.0),
        RandGaussianNoised(keys=keys, prob=0.5, std=0.05),
        RandAdjustContrastd(keys=keys, prob=0.5, gamma=(0.7, 1.3)),
        ToTensord(keys=keys),
    ])
