import numpy as np
import pandas as pd
import warnings
import imutils
import cv2

from typing import List

from os import listdir
from tqdm import tqdm

from ..wdtypes import *

import pdb

def prepare_image(df:pd.DataFrame, img_col:str, img_path:str, width:int,
    height:int, verbose:int=1)->np.ndarray:

    image_list = df[img_col].tolist()

    if verbose: print('Reading Images from {}'.format(img_path))
    imgs = [cv2.imread("/".join([img_path,img])) for img in image_list]

    # finding images with different height and width
    aspect = [(im.shape[0], im.shape[1]) for im in imgs]
    aspect_r = [a[0]/a[1] for a in aspect]
    diff_idx = [i for i,r in enumerate(aspect_r) if r!=1.]

    if verbose: print('Resizing')
    aap = AspectAwarePreprocessor(width, height)
    spp = SimplePreprocessor(width, height)
    resized_imgs = []
    for i,img in tqdm(enumerate(imgs), total=len(imgs), disable=verbose != 1):
        if i in diff_idx:
            resized_imgs.append(aap.preprocess(img))
        else:
            resized_imgs.append(spp.preprocess(img))

    return np.asarray(resized_imgs)


# AspectAwarePreprocessor and SimplePreprocessor are directly taked from the
# great series of Books "Deep Learning for Computer Vision" by Adrian
# (https://www.pyimagesearch.com/author/adrian/). Check here
# https://www.pyimagesearch.com/
class AspectAwarePreprocessor:
    def __init__(self, width:int, height:int, inter=cv2.INTER_AREA):
        self.width = width
        self.height = height
        self.inter = inter

    def preprocess(self, image:np.ndarray)->np.ndarray:
        (h, w) = image.shape[:2]
        dW = 0
        dH = 0

        if w < h:
            image = imutils.resize(image, width=self.width,
                inter=self.inter)
            dH = int((image.shape[0] - self.height) / 2.0)
        else:
            image = imutils.resize(image, height=self.height,
                inter=self.inter)
            dW = int((image.shape[1] - self.width) / 2.0)

        (h, w) = image.shape[:2]
        image = image[dH:h - dH, dW:w - dW]

        return cv2.resize(image, (self.width, self.height),
            interpolation=self.inter)


class SimplePreprocessor:
    def __init__(self, width:int, height:int, inter=cv2.INTER_AREA):
        self.width = width
        self.height = height
        self.inter = inter

    def preprocess(self, image:np.ndarray)->np.ndarray:
        return cv2.resize(image, (self.width, self.height),
            interpolation=self.inter)