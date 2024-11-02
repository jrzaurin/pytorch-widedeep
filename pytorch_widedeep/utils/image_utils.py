"""
AspectAwarePreprocessor and SimplePreprocessor are directly taked from the
great series of Books "Deep Learning for Computer Vision" by Adrian
(https://www.pyimagesearch.com/author/adrian/). Check here
https://www.pyimagesearch.com/

Credit for the code here to ADRIAN ROSEBROCK
"""

import cv2
import numpy as np

__all__ = ["AspectAwarePreprocessor", "SimplePreprocessor"]


# credit for the code here to ADRIAN ROSEBROCK
def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized


class AspectAwarePreprocessor:
    r"""Class to resize an image to a certain width and height taking into account
    the image aspect ratio

    Parameters
    ----------
    width: int
        output width
    height: int
        output height
    inter: interpolation method,  default = ``cv2.INTER_AREA``
        ``opencv`` interpolation method. See ``opencv``
        `InterpolationFlags`.
    """

    def __init__(self, width: int, height: int, inter=cv2.INTER_AREA):
        self.width = width
        self.height = height
        self.inter = inter

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        r"""Returns the resized input image taking into account the image aspect ratio

        Parameters
        ----------
        image: np.ndarray
            Input image to be resized

        Examples
        --------
        >>> import cv2
        >>> from pytorch_widedeep.utils import AspectAwarePreprocessor
        >>> img = cv2.imread("tests/test_data_utils/images/galaxy1.png")
        >>> img.shape
        (694, 890, 3)
        >>> app = AspectAwarePreprocessor(width=224, height=224)
        >>> resized_img = app.preprocess(img)
        >>> resized_img.shape
        (224, 224, 3)

        Returns
        -------
        np.ndarray
            Resized image according to its original image aspect ratio
        """
        (h, w) = image.shape[:2]
        dW = 0
        dH = 0

        if w < h:
            image = resize(image, width=self.width, inter=self.inter)
            dH = int((image.shape[0] - self.height) / 2.0)
        else:
            image = resize(image, height=self.height, inter=self.inter)
            dW = int((image.shape[1] - self.width) / 2.0)

        (h, w) = image.shape[:2]
        image = image[dH : h - dH, dW : w - dW]

        resized_image = cv2.resize(
            image, (self.width, self.height), interpolation=self.inter
        )

        return resized_image


class SimplePreprocessor:
    r"""Class to resize an image to a certain width and height

    Parameters
    ----------
    width: int
        output width
    height: int
        output height
    inter: interpolation method, default = ``cv2.INTER_AREA``
        ``opencv`` interpolation method. See ``opencv``
        `InterpolationFlags`.
    """

    def __init__(self, width: int, height: int, inter=cv2.INTER_AREA):
        self.width = width
        self.height = height
        self.inter = inter

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        r"""Returns the resized input image

        Parameters
        ----------
        image: np.ndarray
            Input image to be resized

        Returns
        -------
        np.ndarray
            Resized image

        """
        resized_image = cv2.resize(
            image, (self.width, self.height), interpolation=self.inter
        )

        return resized_image
