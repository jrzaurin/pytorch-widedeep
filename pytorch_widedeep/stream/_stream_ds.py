from torch.utils.data import IterableDataset

from pytorch_widedeep.utils.fastai_transforms import Vocab
from pytorch_widedeep.stream.preprocessing.text_preprocessor import StreamTextPreprocessor


class StreamTextDataset(IterableDataset):
    def __init__(self, file_path: str, preprocessor: StreamTextPreprocessor, batchsize=1000):
        self.path = file_path
        self.batchsize = batchsize
        self.preprocessor = preprocessor

    def _parse(self):
        # Do a large fetch first - preprocess entire batch and then yield
        batch = []
        ixs = []
        with open(self.path, 'r') as f:
            for ix, line in enumerate(f):
                batch.append(line)
                ixs.append(ix)
                if len(batch) >= 1000:
                    # import pdb; pdb.set_trace()
                    batch = zip(ixs, self.preprocessor.transform(batch))
                    for i, l in batch:
                        yield i, l
                    batch = []
                    ixs = []

    def get_stream(self):
        return self._parse()

    def __iter__(self):
        return self.get_stream()