import glob
from typing import Iterator
import cv2
import pandas as pd
from torch.utils.data import IterableDataset, Dataset

from pytorch_widedeep.utils.fastai_transforms import Vocab
from pytorch_widedeep.stream.preprocessing.text_preprocessor import StreamTextPreprocessor
from pytorch_widedeep.stream.preprocessing.image_preprocessor import StreamImagePreprocessor


class StreamTextDataset(IterableDataset):
    def __init__(
            self, 
            file_path: str, 
            preprocessor: StreamTextPreprocessor, 
            chunksize: int = 1000
        ):
        self.path = file_path
        self.chunksize = chunksize 
        self.preprocessor = preprocessor

    def _process_minibatch(self):
        with pd.read_csv(self.path, chunksize=self.chunksize) as f:
            for batch in f:
                batch = self.preprocessor.transform(batch)
                for ix, row in enumerate(batch):
                    yield ix, row

    def get_stream(self):
        return self._process_minibatch()

    def __iter__(self):
        return self.get_stream()

# Actually doesn't need to stream - we only stream the dataframe and fetch images as needed
class StreamImageDataset(Dataset):
    def __init__(
            self,
            path,
            preprocessor: StreamImagePreprocessor,
            chunksize: int = 1
        ):
        super(StreamImageDataset).__init__()
        self.path = path
        self.preprocessor = preprocessor
        self.chunksize = chunksize # Not sure if we need this yet

    def __getitem__(self, ix: int):
        img = cv2.imread(f"{self.path}/{ix}.*")
        if img.shape[0] != img.shape[1]:
            img = self.preprocessor.aap.preprocess(img)
        else:
            img = self.preprocessor.spp.preprocess(img)
        
        yield 


class StreamWideDeepDataset(IterableDataset):
    def __init__(
            self,
            X_path: str,
            img_col: str,
            text_col: str,
            target_col: str,
            text_preprocessor: StreamTextPreprocessor,
            img_preprocessor: StreamImagePreprocessor,
            chunksize: int = 3
        ):
        super(StreamWideDeepDataset).__init__()
        self.X_path = X_path
        self.img_col = img_col
        self.text_col = text_col
        self.target_col = target_col
        self.text_preprocessor = text_preprocessor
        self.img_preprocessor = img_preprocessor
        self.chunksize = chunksize

    def __iter__(self):
        for chunk in pd.read_csv(self.X_path, chunksize=self.chunksize):
            imgs = self.img_preprocessor.transform(chunk) 
            texts = self.text_preprocessor.transform(chunk)
            target = chunk[self.target_col].values
            yield imgs, texts, target