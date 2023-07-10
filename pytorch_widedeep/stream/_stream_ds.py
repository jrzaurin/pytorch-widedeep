import pandas as pd
from torch.utils.data import IterableDataset

from pytorch_widedeep.utils.fastai_transforms import Vocab
from pytorch_widedeep.stream.preprocessing.text_preprocessor import StreamTextPreprocessor


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