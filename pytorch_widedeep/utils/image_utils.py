'''
AspectAwarePreprocessor and SimplePreprocessor are directly taked from the
great series of Books "Deep Learning for Computer Vision" by Adrian
(https://www.pyimagesearch.com/author/adrian/). Check here
https://www.pyimagesearch.com/'''

import numpy as np
import pandas as pd
import warnings
import imutils
import cv2

from os import listdir
from tqdm import tqdm
from sklearn.utils.validation import check_is_fitted

from .base_util import DataProcessor

from ..wdtypes import *


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


class ImageProcessor(DataProcessor):
    """docstring for ImageProcessor"""
    def __init__(self, width:int=224, height:int=224, verbose:int=1):
        super(ImageProcessor, self).__init__()
        self.width = width
        self.height = height
        self.verbose = verbose

    def fit(self)->DataProcessor:
        self.aap = AspectAwarePreprocessor(self.width, self.height)
        self.spp = SimplePreprocessor(self.width, self.height)
        return self

    def transform(self, df, img_col:str, img_path:str)->np.ndarray:
        check_is_fitted(self, 'aap')
        self.img_col = img_col
        image_list = df[self.img_col].tolist()
        if self.verbose: print('Reading Images from {}'.format(img_path))
        imgs = [cv2.imread("/".join([img_path,img])) for img in image_list]

        # finding images with different height and width
        aspect = [(im.shape[0], im.shape[1]) for im in imgs]
        aspect_r = [a[0]/a[1] for a in aspect]
        diff_idx = [i for i,r in enumerate(aspect_r) if r!=1.]

        if self.verbose: print('Resizing')
        resized_imgs = []
        for i,img in tqdm(enumerate(imgs), total=len(imgs), disable=self.verbose != 1):
            if i in diff_idx:
                resized_imgs.append(self.aap.preprocess(img))
            else:
                resized_imgs.append(self.spp.preprocess(img))

        if self.verbose: print('Computing normalisation metrics')
        mean_R, mean_G, mean_B = [], [], []
        std_R, std_G, std_B = [], [], []
        for rsz_img in resized_imgs:
            (mean_b, mean_g, mean_r), (std_b, std_g, std_r) = cv2.meanStdDev(rsz_img)
            mean_R.append(mean_r), mean_G.append(mean_g), mean_B.append(mean_b)
            std_R.append(std_r), std_G.append(std_g), std_B.append(std_b)
        self.normalise_metrics = dict(
            mean = {"R": np.mean(mean_R)/255., "G": np.mean(mean_G)/255., "B": np.mean(mean_B)/255.},
            std = {"R": np.mean(std_R)/255., "G": np.mean(std_G)/255., "B": np.mean(std_B)/255.}
            )
        return np.asarray(resized_imgs)

    def fit_transform(self, df, img_col:str, img_path:str)->np.ndarray:
        return self.fit().transform(df, img_col, img_path)


