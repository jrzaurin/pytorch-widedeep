import numpy as np
import os

from ..wdtypes import *
from .fastai_transforms import Tokenizer, Vocab
from gensim.utils import tokenize


__all__ = ["simple_preprocess", "get_texts", "pad_sequences", "build_embeddings_matrix"]


def simple_preprocess(
    doc: str,
    lower: bool = False,
    deacc: bool = False,
    min_len: int = 2,
    max_len: int = 15,
) -> List[str]:
    r"""
    Gensim's simple_preprocess adding a 'lower' param to indicate wether or not to
    lower case all the token in the texts

    For more informations see: https://radimrehurek.com/gensim/utils.html
    """
    tokens = [
        token
        for token in tokenize(doc, lower=False, deacc=deacc, errors="ignore")
        if min_len <= len(token) <= max_len and not token.startswith("_")
    ]
    return tokens


def get_texts(texts: List[str]) -> List[List[str]]:
    r"""
    Uses fastai's Tokenizer because it does a series of very convenients things
    during the tokenization process

    See here: https://docs.fast.ai/text.transform.html#Tokenizer
    """
    processed_textx = [" ".join(simple_preprocess(t)) for t in texts]
    tok = Tokenizer().process_all(processed_textx)
    return tok


def pad_sequences(
    seq: List[int], maxlen: int, pad_first: bool = True, pad_idx: int = 1
) -> List[List[int]]:
    r"""
    Given a List of tokenized and 'numericalised' sequences it will return padded sequences
    according to the input parameters maxlen, pad_first and pad_idx

    Parameters
    ----------
    seq: List
        List of int tokens
    maxlen: Int
        Maximum length of the padded sequences
    pad_first: Boolean. Default=True
        Indicates whether the padding index will be added at the beginning or the
        end of the sequences
    pad_idx: Int. Default=1
        padding index. Fastai's Tokenizer leaves 0 for the 'unknown' token.

    Returns:
    res: List
       Padded sequences
    """
    if len(seq) >= maxlen:
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
) -> np.ndarray:
    r"""
    Build the embedding matrix using pretrained word vectors

    Parameters
    ----------
    vocab: Fastai's Vocab object
        see: https://docs.fast.ai/text.transform.html#Vocab
    word_vectors_path:str
        path to the pretrained word embeddings
    min_freq: Int
        minimum frequency required for a word to be in the vocabulary
    verbose: Int. Default=1

    Returns
    -------
    embedding_matrix: np.ndarray
        pretrained word embeddings. If a word in our vocabulary is not among the
        pretrained embeddings it will be assigned the mean pretrained
        word-embeddings vector
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

    mean_word_vector = np.mean(list(embeddings_index.values()), axis=0)
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
