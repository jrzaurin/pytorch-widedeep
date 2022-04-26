"""
AspectAwarePreprocessor and SimplePreprocessor are directly taked from the
great series of Books "Deep Learning for Computer Vision" by Adrian
(https://www.pyimagesearch.com/author/adrian/). Check here
https://www.pyimagesearch.com/

Credit for the code here to ADRIAN ROSEBROCK
"""

import cv2
import numpy as np
import imutils

__all__ = ["AspectAwarePreprocessor", "SimplePreprocessor"]


class AspectAwarePreprocessor:
    r"""Class to resize an image to a certain width and height taking into account
    the image aspect ratio

    Args:
        width (int): output width
        height (int): output height
        inter (interpolation method,  default = ``cv2.INTER_AREA``): ``opencv`` interpolation method. See ``opencv``
            :obj:`InterpolationFlags`.

            .. note:: The value 3 that appears in the class Args: above is a Sphinx
                formatting error.
    """

    def __init__(self, width: int, height: int, inter=cv2.INTER_AREA):
        self.width = width
        self.height = height
        self.inter = inter

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        r"""Returns the resized input image taking into account the image aspect ratio

        Args:
            image (np.ndarray): Input image to be resized

        Examples:
            >>> import cv2
            >>> from pytorch_widedeep.utils import AspectAwarePreprocessor
            >>> img = cv2.imread("tests/test_data_utils/images/galaxy1.png")
            >>> img.shape
            (694, 890, 3)
            >>> app = AspectAwarePreprocessor(width=224, height=224)
            >>> resized_img = app.preprocess(img)
            >>> resized_img.shape
            (224, 224, 3)
        """
        (h, w) = image.shape[:2]
        dW = 0
        dH = 0

        if w < h:
            image = imutils.resize(image, width=self.width, inter=self.inter)
            dH = int((image.shape[0] - self.height) / 2.0)
        else:
            image = imutils.resize(image, height=self.height, inter=self.inter)
            dW = int((image.shape[1] - self.width) / 2.0)

        (h, w) = image.shape[:2]
        image = image[dH : h - dH, dW : w - dW]

        resized_image = cv2.resize(
            image, (self.width, self.height), interpolation=self.inter
        )

        return resized_image


class SimplePreprocessor:
    r"""Class to resize an image to a certain width and height

    Args:
        width (int): output width
        height (int): output height
        inter (interpolation method, default = ``cv2.INTER_AREA``): ``opencv`` interpolation method. See ``opencv``
            :obj:`InterpolationFlags`.

            .. note:: The value 3 that appears in the class Args: above is a Sphinx
                formatting error.
    """

    def __init__(self, width: int, height: int, inter=cv2.INTER_AREA):
        self.width = width
        self.height = height
        self.inter = inter

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        r"""Returns the resized input image

        Args:
            image (np.ndarray): Input image to be resized
        """
        resized_image = cv2.resize(
            image, (self.width, self.height), interpolation=self.inter
        )

        return resized_image
