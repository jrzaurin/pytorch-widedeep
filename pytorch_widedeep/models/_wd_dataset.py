import numpy as np
import torch
from sklearn.utils import Bunch
from torch.utils.data import Dataset

from ..wdtypes import *


class WideDeepDataset(Dataset):
    r"""Dataset object to load WideDeep data to the model

    Parameters
    ----------
    X_wide: np.ndarray, scipy csr sparse matrix.
        wide input.Note that if a sparse matrix is passed to the
        WideDeepDataset class, the loading process will be notably slow since
        the transformation to a dense matrix is done on an index basis 'on the
        fly'. At the moment this is the best option given the current support
        offered for sparse tensors for pytorch.
    X_deep: np.ndarray
        deepdense input
    X_text: np.ndarray
        deeptext input
    X_img: np.ndarray
        deepimage input
    target: np.ndarray
    transforms: MultipleTransforms() object (which is in itself a torchvision
        Compose). See in models/_multiple_transforms.py
    """

    def __init__(
        self,
        X_wide: Union[np.ndarray, sparse_matrix],
        X_deep: np.ndarray,
        target: Optional[np.ndarray] = None,
        X_text: Optional[np.ndarray] = None,
        X_img: Optional[np.ndarray] = None,
        transforms: Optional[Any] = None,
    ):

        self.X_wide = X_wide
        self.X_deep = X_deep
        self.X_text = X_text
        self.X_img = X_img
        self.transforms = transforms
        if self.transforms:
            self.transforms_names = [
                tr.__class__.__name__ for tr in self.transforms.transforms
            ]
        else:
            self.transforms_names = []
        self.Y = target

    def __getitem__(self, idx: int):
        # X_wide and X_deep are assumed to be *always* present
        if isinstance(self.X_wide, sparse_matrix):
            X = Bunch(wide=np.array(self.X_wide[idx].todense()).squeeze())
        else:
            X = Bunch(wide=self.X_wide[idx])
        X.deepdense = self.X_deep[idx]
        if self.X_text is not None:
            X.deeptext = self.X_text[idx]
        if self.X_img is not None:
            # if an image dataset is used, make sure is in the right format to
            # be ingested by the conv layers
            xdi = self.X_img[idx]
            # if int must be uint8
            if "int" in str(xdi.dtype) and "uint8" != str(xdi.dtype):
                xdi = xdi.astype("uint8")
            # if int float must be float32
            if "float" in str(xdi.dtype) and "float32" != str(xdi.dtype):
                xdi = xdi.astype("float32")
            # if there are no transforms, or these do not include ToTensor(),
            # then we need to  replicate what Tensor() does -> transpose axis
            # and normalize if necessary
            if not self.transforms or "ToTensor" not in self.transforms_names:
                xdi = xdi.transpose(2, 0, 1)
                if "int" in str(xdi.dtype):
                    xdi = (xdi / xdi.max()).astype("float32")
            # if ToTensor() is included, simply apply transforms
            if "ToTensor" in self.transforms_names:
                xdi = self.transforms(xdi)
            # else apply transforms on the result of calling torch.tensor on
            # xdi after all the previous manipulation
            elif self.transforms:
                xdi = self.transforms(torch.tensor(xdi))
            # fill the Bunch
            X.deepimage = xdi
        if self.Y is not None:
            y = self.Y[idx]
            return X, y
        else:
            return X

    def __len__(self):
        return len(self.X_deep)
