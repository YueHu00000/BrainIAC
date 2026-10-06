from typing import Any, Dict, List

import torch
from monai.transforms import RandScaleCropd, Resized, Transform


class RandomResizedCrop3D(Transform):
    """
    Combines monai's random spatial crop followed by resize to the desired size.

    Modification:
    1. The spatial crop is done with same dimensions for all the axes
    2. Handles cases where the image_size is less than the crop_size by choosing
        the smallest dimension as the random scale.

    """

    def __init__(self,keys , prob: float = 1.0, size: int = 96, scale: List[float] = [0.2, 1.0]):
        """
        Args:
            scale (List[int]): Specifies the lower and upper bounds for the random area of the crop,
             before resizing. The scale is defined with respect to the area of the original image.
        ** Note Divy : try with random_size=True and random_scale=0.5 (basically just specifying the minimum size of the crop)
        """
        super().__init__()
        self.prob = prob
        self.scale = scale
        self.size = (96,96,96)
        self.keys = keys

    def __call__(self, image):
        if torch.rand(1) < self.prob:
            random_scale = torch.empty(1).uniform_(*self.scale).item()
            rand_cropper = RandScaleCropd(self.keys, random_scale, random_size=False)
            resizer = Resized(self.keys, self.size, mode="trilinear")

            for transform in [rand_cropper, resizer]:
                image = transform(image)

        return image
