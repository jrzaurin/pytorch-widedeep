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
    """Class to resize an image to a certain width and height taking into account
    the image aspect ratio

    Parameters
    ----------
    width: int
        output width
    height: int
        output height
    inter: interpolation method
        ``opencv`` interpolation method. See ``opencv`` ``InterpolationFlags``
    """

    def __init__(self, width: int, height: int, inter=cv2.INTER_AREA):
        self.width = width
        self.height = height
        self.inter = inter

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Resize image

        Parameters
        ----------
        image: np.ndarray

        Returns
        -------
        resized image

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

        return cv2.resize(image, (self.width, self.height), interpolation=self.inter)


class SimplePreprocessor:
    """Class to resize an image to a certain width and height

    Parameters
    ----------
    width: int
        output width
    height: int
        output height
    inter: interpolation method
        ``opencv`` interpolation method. See ``opencv`` ``InterpolationFlags``
    """

    def __init__(self, width: int, height: int, inter=cv2.INTER_AREA):
        self.width = width
        self.height = height
        self.inter = inter

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Resize image

        Parameters
        ----------
        image: np.ndarray

        Returns
        -------
        resized image
        """
        return cv2.resize(image, (self.width, self.height), interpolation=self.inter)
