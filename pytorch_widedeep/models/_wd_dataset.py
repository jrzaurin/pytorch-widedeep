import numpy as np
import torch

from sklearn.utils import Bunch
from torch.utils.data import Dataset

from ..wdtypes import *

class WideDeepDataset(Dataset):
    def __init__(self, X_wide:np.ndarray, X_deep:np.ndarray,
        target:Optional[np.ndarray]=None, X_text:Optional[np.ndarray]=None,
        X_img:Optional[np.ndarray]=None, transforms:Optional=None):

        self.X_wide = X_wide
        self.X_deep = X_deep
        self.X_text = X_text
        self.X_img  = X_img
        self.transforms = transforms
        if self.transforms:
            self.transforms_names = [tr.__class__.__name__ for tr in self.transforms.transforms]
        else: self.transforms_names = []
        self.Y = target

    def __getitem__(self, idx:int):

        X = Bunch(wide=self.X_wide[idx])
        X.deepdense= self.X_deep[idx]
        if self.X_text is not None:
            X.deeptext = self.X_text[idx]
        if self.X_img is not None:
            xdi = self.X_img[idx]
            if 'int' in str(xdi.dtype) and 'uint8' != str(xdi.dtype): xdi = xdi.astype('uint8')
            if 'float' in str(xdi.dtype) and 'float32' != str(xdi.dtype): xdi = xdi.astype('float32')
            if not self.transforms or 'ToTensor' not in self.transforms_names:
                xdi = xdi.transpose(2,0,1)
                if 'int' in str(xdi.dtype): xdi = (xdi/xdi.max()).astype('float32')
            if 'ToTensor' in self.transforms_names: xdi = self.transforms(xdi)
            elif self.transforms: xdi = self.transforms(torch.Tensor(xdi))
            X.deepimage = xdi
        if self.Y is not None:
            y  = self.Y[idx]
            return X, y
        else:
            return X

    def __len__(self):
        return len(self.X_wide)
