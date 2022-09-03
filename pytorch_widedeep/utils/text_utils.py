import os
from typing import List

import numpy as np
from gensim.utils import tokenize

from pytorch_widedeep.utils.fastai_transforms import Vocab, Tokenizer

__all__ = ["simple_preprocess", "get_texts", "pad_sequences", "build_embeddings_matrix"]


def simple_preprocess(
    doc: str,
    lower: bool = False,
    deacc: bool = False,
    min_len: int = 2,
    max_len: int = 15,
) -> List[str]:
    r"""
    This is `Gensim`'s `simple_preprocess` with a `lower` param to
    indicate wether or not to lower case all the token in the doc

    For more information see: `Gensim` [utils module](https://radimrehurek.com/gensim/utils.html)

    Parameters
    ----------
    doc: str
        Input document.
    lower: bool, default = False
        Lower case tokens in the input doc
    deacc: bool, default = False
        Remove accent marks from tokens using `Gensim`'s `deaccent`
    min_len: int, default = 2
        Minimum length of token (inclusive). Shorter tokens are discarded.
    max_len: int, default = 15
        Maximum length of token in result (inclusive). Longer tokens are discarded.

    Examples
    --------
    >>> from pytorch_widedeep.utils import simple_preprocess
    >>> simple_preprocess('Machine learning is great')
    ['Machine', 'learning', 'is', 'great']

    Returns
    -------
    List[str]
        List with the processed tokens
    """
    tokens = [
        token
        for token in tokenize(doc, lower=lower, deacc=deacc, errors="ignore")
        if min_len <= len(token) <= max_len and not token.startswith("_")
    ]
    return tokens


def get_texts(texts: List[str]) -> List[List[str]]:
    r"""Tokenization using `Fastai`'s `Tokenizer` because it does a
    series of very convenients things during the tokenization process

    See `pytorch_widedeep.utils.fastai_utils.Tokenizer`

    Parameters
    ----------
    texts: List
        List of str with the texts (or documents). One str per document

    Examples
    --------
    >>> from pytorch_widedeep.utils import get_texts
    >>> texts = ['Machine learning is great', 'but building stuff is even better']
    >>> get_texts(texts)
    [['xxmaj', 'machine', 'learning', 'is', 'great'], ['but', 'building', 'stuff', 'is', 'even', 'better']]

    Returns
    -------
    List[List[str]]
        List of lists, one list per '_document_' containing its corresponding tokens

    :information_source: **NOTE**:
    `get_texts` uses `pytorch_widedeep.utils.fastai_transforms.Tokenizer`.
    Such tokenizer uses a series of convenient processing steps, including
    the  addition of some special tokens, such as `TK_MAJ` (`xxmaj`), used to
    indicate the next word begins with a capital in the original text. For more
    details of special tokens please see the [`fastai` `docs](https://docs.fast.ai/text.core.html#Tokenizing)
    """
    processed_textx = [" ".join(simple_preprocess(t)) for t in texts]
    tok = Tokenizer().process_all(processed_textx)
    return tok


def pad_sequences(
    seq: List[int], maxlen: int, pad_first: bool = True, pad_idx: int = 1
) -> np.ndarray:
    r"""
    Given a List of tokenized and `numericalised` sequences it will return
    padded sequences according to the input parameters.

    Parameters
    ----------
    seq: List
        List of int with the `numericalised` tokens
    maxlen: int
        Maximum length of the padded sequences
    pad_first: bool,  default = True
        Indicates whether the padding index will be added at the beginning or the
        end of the sequences
    pad_idx: int, default = 1
        padding index. Fastai's Tokenizer leaves 0 for the 'unknown' token.

    Examples
    --------
    >>> from pytorch_widedeep.utils import pad_sequences
    >>> seq = [1,2,3]
    >>> pad_sequences(seq, maxlen=5, pad_idx=0)
    array([0, 0, 1, 2, 3], dtype=int32)

    Returns
    -------
    np.ndarray
        numpy array with the padded sequences
    """
    if len(seq) == 0:
        return np.zeros(maxlen, dtype="int32") + pad_idx
    elif len(seq) >= maxlen:
        res = np.array(seq[-maxlen:]).astype("int32")
        return res
    else:
        res = np.zeros(maxlen, dtype="int32") + pad_idx
        if pad_first:
            res[-len(seq) :] = seq
        else:
            res[: len(seq) :] = seq
        return res


def build_embeddings_matrix(
    vocab: Vocab, word_vectors_path: str, min_freq: int, verbose: int = 1
) -> np.ndarray:  # pragma: no cover
    r"""Build the embedding matrix using pretrained word vectors.

    Returns pretrained word embeddings. If a word in our vocabulary is not
    among the pretrained embeddings it will be assigned the mean pretrained
    word-embeddings vector

    Parameters
    ----------
    vocab: Vocab
        see `pytorch_widedeep.utils.fastai_utils.Vocab`
    word_vectors_path: str
        path to the pretrained word embeddings
    min_freq: int
        minimum frequency required for a word to be in the vocabulary
    verbose: int,  default=1
        level of verbosity. Set to 0 for no verbosity

    Returns
    -------
    np.ndarray
        Pretrained word embeddings
    """
    if not os.path.isfile(word_vectors_path):
        raise FileNotFoundError("{} not found".format(word_vectors_path))
    if verbose:
        print("Indexing word vectors...")

    embeddings_index = {}
    f = open(word_vectors_path)
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype="float32")
        embeddings_index[word] = coefs
    f.close()

    if verbose:
        print("Loaded {} word vectors".format(len(embeddings_index)))
        print("Preparing embeddings matrix...")

    mean_word_vector = np.mean(list(embeddings_index.values()), axis=0)  # type: ignore[arg-type]
    embedding_dim = len(list(embeddings_index.values())[0])
    num_words = len(vocab.itos)
    embedding_matrix = np.zeros((num_words, embedding_dim))
    found_words = 0
    for i, word in enumerate(vocab.itos):
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
            found_words += 1
        else:
            embedding_matrix[i] = mean_word_vector

    if verbose:
        print(
            "{} words in the vocabulary had {} vectors and appear more than {} times".format(
                found_words, word_vectors_path, min_freq
            )
        )

    return embedding_matrix.astype("float32")
