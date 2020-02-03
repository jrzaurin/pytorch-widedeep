"""
AspectAwarePreprocessor and SimplePreprocessor are directly taked from the
great series of Books "Deep Learning for Computer Vision" by Adrian
(https://www.pyimagesearch.com/author/adrian/). Check here
https://www.pyimagesearch.com/

Credit for the code here to ADRIAN ROSEBROCK
"""

import numpy as np
import imutils
import cv2


__all__ = ["AspectAwarePreprocessor", "SimplePreprocessor"]


class AspectAwarePreprocessor:
    def __init__(self, width: int, height: int, inter=cv2.INTER_AREA):
        self.width = width
        self.height = height
        self.inter = inter

    def preprocess(self, image: np.ndarray) -> np.ndarray:
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
    def __init__(self, width: int, height: int, inter=cv2.INTER_AREA):
        self.width = width
        self.height = height
        self.inter = inter

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        return cv2.resize(image, (self.width, self.height), interpolation=self.inter)
