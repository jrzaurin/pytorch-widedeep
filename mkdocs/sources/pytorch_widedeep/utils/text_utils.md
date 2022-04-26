#


### simple_preprocess
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/utils/text_utils.py/#L12)
```python
.simple_preprocess(
   doc: str, lower: bool = False, deacc: bool = False, min_len: int = 2, max_len: int = 15
)
```

---
This is ``Gensim``'s :obj:`simple_preprocess` with a ``lower`` param to
indicate wether or not to lower case all the token in the doc

For more information see: ``Gensim`` `utils module
<https://radimrehurek.com/gensim/utils.html>`_. Returns the list of tokens
for a given doc

Parameters
----------
doc: str
Input document.
---
    Lower case tokens in the input doc
    Remove accent marks from tokens using ``Gensim``'s :obj:`deaccent`
    Minimum length of token (inclusive). Shorter tokens are discarded.
    Maximum length of token in result (inclusive). Longer tokens are discarded.

Examples
--------

```python

>>> simple_preprocess('Machine learning is great')
['Machine', 'learning', 'is', 'great']
```

----


### get_texts
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/utils/text_utils.py/#L54)
```python
.get_texts(
   texts: List[str]
)
```

---
Tokenization using ``Fastai``'s :obj:`Tokenizer` because it does a
series of very convenients things during the tokenization process

See :class:`pytorch_widedeep.utils.fastai_utils.Tokenizer`

Returns a list containing the tokens per text or document

Parameters
----------
texts: List
List of str with the texts (or documents). One str per document

---
Examples
--------

```python

>>> texts = ['Machine learning is great', 'but building stuff is even better']
>>> get_texts(texts)
[['xxmaj', 'machine', 'learning', 'is', 'great'], ['but', 'building', 'stuff', 'is', 'even', 'better']]

```
    <https://docs.fast.ai/text.transform.html#Tokenizer>`_.

----


### pad_sequences
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/utils/text_utils.py/#L86)
```python
.pad_sequences(
   seq: List[int], maxlen: int, pad_first: bool = True, pad_idx: int = 1
)
```

---
Given a List of tokenized and `numericalised` sequences it will return
padded sequences according to the input parameters.

Parameters
----------
seq: List
List of int with the `numericalised` tokens
---
    Maximum length of the padded sequences
    end of the sequences
    padding index. Fastai's Tokenizer leaves 0 for the 'unknown' token.

Examples
--------

```python

>>> seq = [1,2,3]
>>> pad_sequences(seq, maxlen=5, pad_idx=0)
array([0, 0, 1, 2, 3], dtype=int32)
```

----


### build_embeddings_matrix
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/utils/text_utils.py/#L126)
```python
.build_embeddings_matrix(
   vocab: Vocab, word_vectors_path: str, min_freq: int, verbose: int = 1
)
```

---
Build the embedding matrix using pretrained word vectors. Returns the
pretrained word embeddings

Returns pretrained word embeddings. If a word in our vocabulary is not
among the pretrained embeddings it will be assigned the mean pretrained
word-embeddings vector

Parameters
----------
vocab: Vocab
see :class:`pytorch_widedeep.utils.fastai_utils.Vocab`
---
    path to the pretrained word embeddings
    minimum frequency required for a word to be in the vocabulary
    level of verbosity. Set to 0 for no verbosity
