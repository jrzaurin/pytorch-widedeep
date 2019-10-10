import numpy as np
import pandas as pd
import html
import os
import re

from pathlib import PosixPath
from typing import List
from gensim.utils import tokenize
from sklearn.utils.validation import check_is_fitted

from .fastai_transforms import Tokenizer, Vocab
from .base_util import DataProcessor
from ..wdtypes import *


def simple_preprocess(doc:str, lower:bool=False, deacc:bool=False, min_len:int=2,
	max_len:int=15) -> List[str]:
    tokens = [
        token for token in tokenize(doc, lower=False, deacc=deacc, errors='ignore')
        if min_len <= len(token) <= max_len and not token.startswith('_')
    ]
    return tokens


def get_texts(texts:List[str]) -> List[List[str]]:
    processed_textx = [' '.join(simple_preprocess(t)) for t in texts]
    tok = Tokenizer().process_all(processed_textx)
    return tok


def pad_sequences(seq:List[int], maxlen:int=190, pad_first:bool=True, pad_idx:int=1) -> List[List[int]]:
    if len(seq) >= maxlen:
        res = np.array(seq[-maxlen:]).astype('int32')
        return res
    else:
        res = np.zeros(maxlen, dtype='int32') + pad_idx
        if pad_first: res[-len(seq):] = seq
        else:         res[:len(seq):] = seq
        return res


def build_embeddings_matrix(vocab:Vocab, word_vectors_path:str, verbose:int=1) -> np.ndarray:

	if not os.path.isfile(word_vectors_path):
		raise FileNotFoundError("{} not found".format(word_vectors_path))
	if verbose: print('Indexing word vectors...')

	embeddings_index = {}
	f = open(word_vectors_path)
	for line in f:
	    values = line.split()
	    word = values[0]
	    coefs = np.asarray(values[1:], dtype='float32')
	    embeddings_index[word] = coefs
	f.close()

	if verbose:
		print('Loaded {} word vectors'.format(len(embeddings_index)))
		print('Preparing embeddings matrix...')

	mean_word_vector = np.mean(list(embeddings_index.values()), axis=0)
	embedding_dim = len(list(embeddings_index.values())[0])
	num_words = len(vocab.itos)
	embedding_matrix = np.zeros((num_words, embedding_dim))
	found_words=0
	for i,word in enumerate(vocab.itos):
	    embedding_vector = embeddings_index.get(word)
	    if embedding_vector is not None:
	        embedding_matrix[i] = embedding_vector
	        found_words+=1
	    else:
	        embedding_matrix[i] = mean_word_vector

	if verbose:
		print('{} words in the vocabulary had {} vectors and appear more than the min frequency'.format(found_words, word_vectors_path))

	return embedding_matrix


class TextProcessor(DataProcessor):
	"""docstring for TextProcessor"""
	def __init__(self, max_vocab:int=30000, min_freq:int=5,
		maxlen:int=80, word_vectors_path:Optional[str]=None,
		verbose:int=1):
		super(TextProcessor, self).__init__()
		self.max_vocab = max_vocab
		self.min_freq = min_freq
		self.maxlen = maxlen
		self.word_vectors_path = word_vectors_path
		self.verbose = verbose

	def fit(self, df:pd.DataFrame, text_col:str)->DataProcessor:
		text_col = text_col
		texts = df[text_col].tolist()
		tokens = get_texts(texts)
		self.vocab = Vocab.create(tokens, max_vocab=self.max_vocab, min_freq=self.min_freq)
		return self

	def transform(self, df:pd.DataFrame, text_col:str)->np.ndarray:
		check_is_fitted(self, 'vocab')
		self.text_col = text_col
		texts = df[self.text_col].tolist()
		self.tokens = get_texts(texts)
		sequences = [self.vocab.numericalize(t) for t in self.tokens]
		padded_seq = np.array([pad_sequences(s, maxlen=self.maxlen) for s in sequences])
		if self.verbose:
		    print("The vocabulary contains {} words".format(len(self.vocab.stoi)))
		if self.word_vectors_path is not None:
		    self.embedding_matrix = build_embeddings_matrix(self.vocab, self.word_vectors_path)
		return padded_seq

	def fit_transform(self, df:pd.DataFrame, text_col:str)->np.ndarray:
		return self.fit(df, text_col).transform(df, text_col)


