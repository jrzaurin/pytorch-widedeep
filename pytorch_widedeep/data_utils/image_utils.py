import numpy as np
import pandas as pd
import imutils
import cv2

from typing import List

from os import listdir
from tqdm import tqdm
from pathlib import Path
from pathlib import PosixPath

from ..wdtypes import *

def prepare_image(df:pd.DataFrame, img_col:str, img_path:PosixPath, width:int,
    height:int, verbose:int=1)->np.ndarray:

    imgnames = listdir(img_path)
    ids_with_images = [int(s.split('.')[0]) for s in imgnames]

    # ids with no image file
    assert df[img_col].nunique() <= len(ids_with_images), 'Read {} images but got {} unique image ids in the dataframe'.format(
        len(ids_with_images), df[img_col].unique())

    # Make sure the order of the images is the same as the rest of the
    # features and text
    image_list = [str(i)+'.jpg' for i in df[img_col]]

    if verbose: print('Reading Images from {}'.format(img_path))
    imgs = [cv2.imread(str(img_path/img)) for img in image_list]

    # finding images with different height and width
    aspect = [(im.shape[0], im.shape[1]) for im in imgs]
    aspect_r = [a[0]/a[1] for a in aspect]
    diff_idx = [i for i,r in enumerate(aspect_r) if r!=1.]

    if verbose: print('Resizing')
    aap = AspectAwarePreprocessor(width, height)
    spp = SimplePreprocessor(width, height)
    resized_imgs = []
    for i,img in tqdm(enumerate(imgs), total=len(imgs)):
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






