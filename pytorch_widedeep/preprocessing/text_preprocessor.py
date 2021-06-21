import numpy as np
import pandas as pd

from pytorch_widedeep.wdtypes import *  # noqa: F403
from pytorch_widedeep.utils.text_utils import (
    get_texts,
    pad_sequences,
    build_embeddings_matrix,
)
from pytorch_widedeep.utils.fastai_transforms import Vocab
from pytorch_widedeep.preprocessing.base_preprocessor import (
    BasePreprocessor,
    check_is_fitted,
)


class TextPreprocessor(BasePreprocessor):
    r"""Preprocessor to prepare the ``deeptext`` input dataset

    Parameters
    ----------
    text_col: str
        column in the input dataframe containing the texts
    max_vocab: int, default=30000
        Maximum number of tokens in the vocabulary
    min_freq: int, default=5
        Minimum frequency for a token to be part of the vocabulary
    maxlen: int, default=80
        Maximum length of the tokenized sequences
    pad_first: bool,  default = True
        Indicates whether the padding index will be added at the beginning or the
        end of the sequences
    pad_idx: int, default = 1
        padding index. Fastai's Tokenizer leaves 0 for the 'unknown' token.
    word_vectors_path: str, Optional
        Path to the pretrained word vectors
    verbose: int, default 1
        Enable verbose output.

    Attributes
    ----------
    vocab: Vocab
        an instance of :class:`pytorch_widedeep.utils.fastai_transforms.Vocab`
    embedding_matrix: np.ndarray
        Array with the pretrained embeddings
    tokens: List
        List with Lists of str containing the tokenized texts

    Examples
    ---------
    >>> import pandas as pd
    >>> from pytorch_widedeep.preprocessing import TextPreprocessor
    >>> df_train = pd.DataFrame({'text_column': ["life is like a box of chocolates",
    ... "You never know what you're gonna get"]})
    >>> text_preprocessor = TextPreprocessor(text_col='text_column', max_vocab=25, min_freq=1, maxlen=10)
    >>> text_preprocessor.fit_transform(df_train)
    The vocabulary contains 24 tokens
    array([[ 1,  1,  1,  1, 10, 11, 12, 13, 14, 15],
           [ 5,  9, 16, 17, 18,  9, 19, 20, 21, 22]], dtype=int32)
    >>> df_te = pd.DataFrame({'text_column': ['you never know what is in the box']})
    >>> text_preprocessor.transform(df_te)
    array([[ 1,  1,  9, 16, 17, 18, 11,  0,  0, 13]], dtype=int32)
    """

    def __init__(
        self,
        text_col: str,
        max_vocab: int = 30000,
        min_freq: int = 5,
        maxlen: int = 80,
        pad_first: bool = True,
        pad_idx: int = 1,
        word_vectors_path: Optional[str] = None,
        verbose: int = 1,
    ):
        super(TextPreprocessor, self).__init__()

        self.text_col = text_col
        self.max_vocab = max_vocab
        self.min_freq = min_freq
        self.maxlen = maxlen
        self.pad_first = pad_first
        self.pad_idx = pad_idx
        self.word_vectors_path = word_vectors_path
        self.verbose = verbose

    def fit(self, df: pd.DataFrame) -> BasePreprocessor:
        """Builds the vocabulary"""
        texts = df[self.text_col].tolist()
        tokens = get_texts(texts)
        self.vocab = Vocab.create(
            tokens, max_vocab=self.max_vocab, min_freq=self.min_freq
        )
        if self.verbose:
            print("The vocabulary contains {} tokens".format(len(self.vocab.stoi)))
        if self.word_vectors_path is not None:
            self.embedding_matrix = build_embeddings_matrix(
                self.vocab, self.word_vectors_path, self.min_freq
            )
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Returns the padded, `numericalised` sequences"""
        check_is_fitted(self, attributes=["vocab"])
        texts = df[self.text_col].tolist()
        self.tokens = get_texts(texts)
        sequences = [self.vocab.numericalize(t) for t in self.tokens]
        padded_seq = np.array(
            [
                pad_sequences(
                    s,
                    maxlen=self.maxlen,
                    pad_first=self.pad_first,
                    pad_idx=self.pad_idx,
                )
                for s in sequences
            ]
        )
        return padded_seq

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """Combines ``fit`` and ``transform``"""
        return self.fit(df).transform(df)

    def inverse_transform(self, padded_seq: np.ndarray) -> pd.DataFrame:
        """Returns the original text plus the added 'special' tokens"""
        texts = [self.vocab.textify(num) for num in padded_seq]
        return pd.DataFrame({self.text_col: texts})
