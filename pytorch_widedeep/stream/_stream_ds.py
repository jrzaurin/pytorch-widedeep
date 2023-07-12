from itertools import repeat
import torch
import pandas as pd
from sklearn.utils import Bunch
from torch.utils.data import IterableDataset

from pytorch_widedeep.stream.preprocessing.text_preprocessor import StreamTextPreprocessor
from pytorch_widedeep.stream.preprocessing.image_preprocessor import StreamImagePreprocessor


class StreamWideDeepDataset(IterableDataset):
    def __init__(
            self,
            X_path: str,
            target_col: str,
            img_col: str = None,
            text_col: str = None,
            text_preprocessor: StreamTextPreprocessor = None,
            img_preprocessor: StreamImagePreprocessor = None,
            transforms = None,
            fetch_size: int = 64
        ):
        super(StreamWideDeepDataset).__init__()
        self.X_path = X_path
        self.img_col = img_col
        self.text_col = text_col
        self.target_col = target_col
        self.text_preprocessor = text_preprocessor
        self.img_preprocessor = img_preprocessor
        self.fetch_size = fetch_size

        # TODO: Fix transforms
        self.transforms = transforms
        self.transforms_names = [None]

        img_preprocessor.verbose = False

    def __iter__(self):
        # Make this also be able to yield rows from an in-memory X
        for chunk in pd.read_csv(self.X_path, chunksize=self.fetch_size):
            text = repeat(None, self.fetch_size)
            imgs = repeat(None, self.fetch_size)
            if self.text_col:
                text = self.text_preprocessor.transform(chunk)
            if self.img_col:
                imgs = self.img_preprocessor.transform(chunk)  # TODO: add prepare images

            target = chunk[self.target_col].values
            for im, txt, tar in zip(imgs, text, target):
                x = Bunch()
                if self.img_col:
                    x.deepimage = self._prepare_image(im)
                if self.text_col:
                    x.deeptext = txt
                yield x, tar 

    def _prepare_image(self, xdi: torch.tensor):
        # if an image dataset is used, make sure is in the right format to
        # be ingested by the conv layers
        # xdi = self.X_img[idx]
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
            if xdi.ndim == 2:
                xdi = xdi[:, :, None]
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
        return xdi