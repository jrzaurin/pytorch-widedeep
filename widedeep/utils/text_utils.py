import numpy as np
import html
import re

from typing import List
from gensim.utils import tokenize
from fastai.text import Tokenizer
from fastai.text.transform import Vocab


def simple_preprocess(doc:str, lower:bool=False, deacc:bool=False, min_len:int=2,
	max_len:int=15) -> List[str]:
    tokens = [
        token for token in tokenize(doc, lower=False, deacc=deacc, errors='ignore')
        if min_len <= len(token) <= max_len and not token.startswith('_')
    ]
    return tokens


def fix_html(text:str) -> str:
	"""
	helper taken from the fastai course 2018 lesson 3:
	Check here: http://course18.fast.ai/
	"""
	re1 = re.compile(r'  +')
	text = text.replace('#39;', "'").replace('amp;', '&').replace('#146;', "'").replace(
	    'nbsp;', ' ').replace('#36;', '$').replace('\\n', "\n").replace('quot;', "'").replace(
	    '<br />', "\n").replace('\\"', '"').replace('<unk>','u_n').replace(' @.@ ','.').replace(
	    ' @-@ ','-').replace('\\', ' \\ ')
	return re1.sub(' ', html.unescape(text))


def get_texts_fastai(texts:List[str]) -> List[List[str]]:
    fixed_texts = [fix_html(t) for t in texts]
    tok = Tokenizer().process_all(fixed_texts)
    return tok


def get_texts_gensim(texts:List[str]) -> List[List[str]]:
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


def build_embeddings_matrix(vocab:Vocab, word_vectors_file:str) -> np.ndarray:

	if 'fasttext' in word_vectors_file:
		word_vectors = 'fasttext'
	elif 'glove' in word_vectors_file:
		word_vectors = 'glove'

	print('Indexing word vectors...')
	embeddings_index = {}
	f = open(word_vectors_file)
	for line in f:
	    values = line.split()
	    word = values[0]
	    coefs = np.asarray(values[1:], dtype='float32')
	    embeddings_index[word] = coefs
	f.close()
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
	print('{} words in our vocabulary had {} vectors and appear more than the min frequency'.format(found_words, word_vectors))

	return embedding_matrix
