import os
from typing import Optional
from collections import Counter, defaultdict

import pandas as pd
import numpy as np

from pytorch_widedeep.preprocessing.base_preprocessor import BasePreprocessor, check_is_fitted

from pytorch_widedeep.preprocessing import TextPreprocessor
from pytorch_widedeep.utils.fastai_transforms import Tokens, Vocab, defaults
from pytorch_widedeep.utils.text_utils import get_texts, build_embeddings_matrix, pad_sequences


class VocabBuilder:
    def __init__(self, max_vocab: int = 30000, min_freq: int = 5) -> None:
        self.freq = Counter()
        self.itos = None 
        self.stoi = None
        self.max_vocab = max_vocab
        self.min_freq = min_freq

    def ingest(self, tokens: Tokens) -> None:
        tokens = [p for o in tokens for p in o]
        for t in tokens:
            self.freq[t] += 1
        
    def build(self) -> Vocab:
        itos = [o for o, c in self.freq.most_common(self.max_vocab) if c >= self.min_freq]
        for o in reversed(defaults.text_spec_tok):
            if o in itos:
                itos.remove(o)
            itos.insert(0, o)
        itos = itos[:self.max_vocab]
        if (
            len(itos) < self.max_vocab
        ):  # Make sure vocab size is a multiple of 8 for fast mixed precision training
            while len(itos) % 8 != 0:
                itos.append("xxfake")

        self.itos = itos
        self.stoi = defaultdict(int, {v: k for k, v in enumerate(itos)})

        return Vocab(self.itos)


class StreamTextPreprocessor(TextPreprocessor):
    def __init__(
        self,
        text_col: str,
        max_vocab: int = 30000,
        min_freq: int = 5,
        maxlen: int = 80,
        pad_first: bool = True,
        pad_idx: int = 1,
        word_vectors_path: Optional[str] = None,
        n_cpus: Optional[int] = None,
        verbose: int = 1,
    ):
        super(StreamTextPreprocessor, self).__init__(text_col)

        self.text_col = text_col
        self.max_vocab = max_vocab
        self.min_freq = min_freq
        self.maxlen = maxlen
        self.pad_first = pad_first
        self.pad_idx = pad_idx
        self.word_vectors_path = word_vectors_path
        self.verbose = verbose
        self.n_cpus = n_cpus if n_cpus is not None else os.cpu_count()

    def fit(self, X_path: str, chunksize: int) -> BasePreprocessor:
        voc_builder = VocabBuilder(max_vocab=self.max_vocab, min_freq=self.min_freq)

        for chunk in pd.read_csv(X_path, chunksize=chunksize):
            tokens = get_texts(chunk[self.text_col].tolist())
            voc_builder.ingest(tokens)

        # BUG: padding is always being applied until max sequence length - error with mix-precision somewhere? 
        self.vocab = voc_builder.build()   

        if self.verbose:
            print("The vocabulary contains {} tokens".format(len(self.vocab.stoi)))

        return self

    def fit_transform(self):
        raise NotImplementedError('''
            fit_transform is not applicable for a StreamTextPreprocessor.
            Fit should be called prior to passing the processor into the StreamTrainer.
        ''')