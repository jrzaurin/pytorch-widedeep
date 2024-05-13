import os
from typing import List, Union, Optional

import numpy as np
import pandas as pd

from pytorch_widedeep.utils.text_utils import (
    get_texts,
    pad_sequences,
    build_embeddings_matrix,
)
from pytorch_widedeep.utils.general_utils import alias
from pytorch_widedeep.utils.fastai_transforms import Vocab, ChunkVocab
from pytorch_widedeep.preprocessing.base_preprocessor import (
    BasePreprocessor,
    check_is_fitted,
)

TVocab = Union[Vocab, ChunkVocab]


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
    already_processed: bool, Optional, default = False
        Boolean indicating if the sequence of elements is already processed or
        prepared. If this is the case, this Preprocessor will simply tokenize
        and pad the sequence. <br/>

            Param aliases: `not_text`. <br/>

        This parameter is thought for those cases where the input sequences
        are already fully processed or are directly not text (e.g. IDs)
    word_vectors_path: str, Optional
        Path to the pretrained word vectors
    n_cpus: int, Optional, default = None
        number of CPUs to used during the tokenization process
    verbose: int, default 1
        Enable verbose output.

    Attributes
    ----------
    vocab: Vocab
        an instance of `pytorch_widedeep.utils.fastai_transforms.Vocab`
    embedding_matrix: np.ndarray
        Array with the pretrained embeddings

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

    @alias("already_processed", ["not_text"])
    def __init__(
        self,
        text_col: str,
        max_vocab: int = 30000,
        min_freq: int = 5,
        maxlen: int = 80,
        pad_first: bool = True,
        pad_idx: int = 1,
        already_processed: Optional[bool] = False,
        word_vectors_path: Optional[str] = None,
        n_cpus: Optional[int] = None,
        verbose: int = 1,
    ):
        super(TextPreprocessor, self).__init__()

        self.text_col = text_col
        self.max_vocab = max_vocab
        self.min_freq = min_freq
        self.maxlen = maxlen
        self.pad_first = pad_first
        self.pad_idx = pad_idx
        self.already_processed = already_processed
        self.word_vectors_path = word_vectors_path
        self.verbose = verbose
        self.n_cpus = n_cpus if n_cpus is not None else os.cpu_count()

        self.is_fitted = False

    def fit(self, df: pd.DataFrame) -> BasePreprocessor:
        """Builds the vocabulary

        Parameters
        ----------
        df: pd.DataFrame
            Input pandas dataframe

        Returns
        -------
        TextPreprocessor
            `TextPreprocessor` fitted object
        """
        texts = self._read_texts(df)

        tokens = get_texts(texts, self.already_processed, self.n_cpus)

        self.vocab: TVocab = Vocab(
            max_vocab=self.max_vocab,
            min_freq=self.min_freq,
            pad_idx=self.pad_idx,
        ).fit(
            tokens,
        )

        if self.verbose:
            print("The vocabulary contains {} tokens".format(len(self.vocab.stoi)))
        if self.word_vectors_path is not None:
            self.embedding_matrix = build_embeddings_matrix(
                self.vocab, self.word_vectors_path, self.min_freq
            )

        self.is_fitted = True

        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Returns the padded, _'numericalised'_ sequences

        Parameters
        ----------
        df: pd.DataFrame
            Input pandas dataframe

        Returns
        -------
        np.ndarray
            Padded, _'numericalised'_ sequences
        """
        check_is_fitted(self, attributes=["vocab"])
        texts = self._read_texts(df)
        tokens = get_texts(texts, self.already_processed, self.n_cpus)
        return self._pad_sequences(tokens)

    def transform_sample(self, text: str) -> np.ndarray:
        """Returns the padded, _'numericalised'_ sequence

        Parameters
        ----------
        text: str
            text to be tokenized and padded

        Returns
        -------
        np.ndarray
            Padded, _'numericalised'_ sequence
        """
        check_is_fitted(self, attributes=["vocab"])
        tokens = get_texts([text], self.already_processed, self.n_cpus)
        return self._pad_sequences(tokens)[0]

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """Combines `fit` and `transform`

        Parameters
        ----------
        df: pd.DataFrame
            Input pandas dataframe

        Returns
        -------
        np.ndarray
            Padded, _'numericalised'_ sequences
        """
        return self.fit(df).transform(df)

    def inverse_transform(self, padded_seq: np.ndarray) -> pd.DataFrame:
        """Returns the original text plus the added 'special' tokens

        Parameters
        ----------
        padded_seq: np.ndarray
            array with the output of the `transform` method

        Returns
        -------
        pd.DataFrame
            Pandas dataframe with the original text plus the added 'special' tokens
        """
        texts = [self.vocab.inverse_transform(num) for num in padded_seq]
        return pd.DataFrame({self.text_col: texts})

    def _pad_sequences(self, tokens: List[List[str]]) -> np.ndarray:
        sequences = [self.vocab.transform(t) for t in tokens]
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

    def _read_texts(
        self, df: pd.DataFrame, root_dir: Optional[str] = None
    ) -> List[str]:
        if root_dir is not None:
            if not os.path.exists(root_dir):
                raise ValueError(
                    "root_dir does not exist. Please create it before fitting the preprocessor"
                )
            texts_fnames = df[self.text_col].tolist()
            texts: List[str] = []
            for texts_fname in texts_fnames:
                with open(os.path.join(root_dir, texts_fname), "r") as f:
                    texts.append(f.read().replace("\n", ""))
        else:
            texts = df[self.text_col].tolist()

        return texts

    def _load_vocab(self, vocab: TVocab) -> None:
        self.vocab = vocab

    def __repr__(self) -> str:
        list_of_params: List[str] = ["text_col={text_col}"]
        list_of_params.append("max_vocab={max_vocab}")
        list_of_params.append("min_freq={min_freq}")
        list_of_params.append("maxlen={maxlen}")
        list_of_params.append("pad_first={pad_first}")
        list_of_params.append("pad_idx={pad_idx}")
        list_of_params.append("already_processed={already_processed}")
        if self.word_vectors_path is not None:
            list_of_params.append("word_vectors_path={word_vectors_path}")
        if self.n_cpus is not None:
            list_of_params.append("n_cpus={n_cpus}")
        if self.verbose is not None:
            list_of_params.append("verbose={verbose}")
        all_params = ", ".join(list_of_params)
        return f"TextPreprocessor({all_params.format(**self.__dict__)})"


class ChunkTextPreprocessor(TextPreprocessor):
    r"""Preprocessor to prepare the ``deeptext`` input dataset

    Parameters
    ----------
    text_col: str
        column in the input dataframe containing either the texts or the
        filenames where the text documents are stored
    n_chunks: int
        Number of chunks that the text dataset is divided by.
    root_dir: str, Optional, default = None
        If 'text_col' contains the filenames with the text documents, this is
        the path to the directory where those documents are stored.
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
    n_cpus: int, Optional, default = None
        number of CPUs to used during the tokenization process
    verbose: int, default 1
        Enable verbose output.

    Attributes
    ----------
    vocab: Vocab
        an instance of `pytorch_widedeep.utils.fastai_transforms.ChunkVocab`
    embedding_matrix: np.ndarray
        Array with the pretrained embeddings if `word_vectors_path` is not None

    Examples
    ---------
    >>> import pandas as pd
    >>> from pytorch_widedeep.preprocessing import ChunkTextPreprocessor
    >>> chunk_df = pd.DataFrame({'text_column': ["life is like a box of chocolates",
    ... "You never know what you're gonna get"]})
    >>> chunk_text_preprocessor = ChunkTextPreprocessor(text_col='text_column', n_chunks=1,
    ... max_vocab=25, min_freq=1, maxlen=10, verbose=0, n_cpus=1)
    >>> processed_chunk = chunk_text_preprocessor.fit_transform(chunk_df)
    """

    def __init__(
        self,
        text_col: str,
        n_chunks: int,
        root_dir: Optional[str] = None,
        max_vocab: int = 30000,
        min_freq: int = 5,
        maxlen: int = 80,
        pad_first: bool = True,
        pad_idx: int = 1,
        already_processed: Optional[bool] = False,
        word_vectors_path: Optional[str] = None,
        n_cpus: Optional[int] = None,
        verbose: int = 1,
    ):
        super(ChunkTextPreprocessor, self).__init__(
            text_col=text_col,
            max_vocab=max_vocab,
            min_freq=min_freq,
            maxlen=maxlen,
            pad_first=pad_first,
            pad_idx=pad_idx,
            already_processed=already_processed,
            word_vectors_path=word_vectors_path,
            n_cpus=n_cpus,
            verbose=verbose,
        )

        self.n_chunks = n_chunks
        self.root_dir = root_dir

        self.chunk_counter = 0

        self.is_fitted = False

    def partial_fit(self, df: pd.DataFrame) -> "ChunkTextPreprocessor":
        # df is a chunk of the original dataframe
        self.chunk_counter += 1

        texts = self._read_texts(df, self.root_dir)

        tokens = get_texts(texts, self.already_processed, self.n_cpus)

        if not hasattr(self, "vocab"):
            self.vocab = ChunkVocab(
                max_vocab=self.max_vocab,
                min_freq=self.min_freq,
                pad_idx=self.pad_idx,
                n_chunks=self.n_chunks,
            )

        self.vocab.fit(tokens)

        if self.chunk_counter == self.n_chunks:
            if self.verbose:
                print("The vocabulary contains {} tokens".format(len(self.vocab.stoi)))
            if self.word_vectors_path is not None:
                self.embedding_matrix = build_embeddings_matrix(
                    self.vocab, self.word_vectors_path, self.min_freq
                )

            self.is_fitted = True

        return self

    def fit(self, df: pd.DataFrame) -> "ChunkTextPreprocessor":
        # df is a chunk of the original dataframe
        return self.partial_fit(df)

    def __repr__(self) -> str:
        list_of_params: List[str] = ["text_col='{text_col}'"]
        if self.n_chunks is not None:
            list_of_params.append("n_chunks={n_chunks}")
        if self.root_dir is not None:
            list_of_params.append("root_dir={root_dir}")
        list_of_params.append("max_vocab={max_vocab}")
        list_of_params.append("min_freq={min_freq}")
        list_of_params.append("maxlen={maxlen}")
        list_of_params.append("pad_first={pad_first}")
        list_of_params.append("pad_idx={pad_idx}")
        if self.word_vectors_path is not None:
            list_of_params.append("word_vectors_path={word_vectors_path}")
        if self.n_cpus is not None:
            list_of_params.append("n_cpus={n_cpus}")
        list_of_params.append("verbose={verbose}")
        all_params = ", ".join(list_of_params)
        return f"ChunkTextPreprocessor({all_params.format(**self.__dict__)})"
