#


## TextPreprocessor
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/preprocessing/text_preprocessor.py/#L17)
```python 
TextPreprocessor(
   text_col: str, max_vocab: int = 30000, min_freq: int = 5, maxlen: int = 80,
   pad_first: bool = True, pad_idx: int = 1, word_vectors_path: Optional[str] = None,
   verbose: int = 1
)
```


---
Preprocessor to prepare the ``deeptext`` input dataset

Parameters
----------
text_col: str
column in the input dataframe containing the texts
---
    Maximum number of tokens in the vocabulary
    Minimum frequency for a token to be part of the vocabulary
    Maximum length of the tokenized sequences
    end of the sequences
    padding index. Fastai's Tokenizer leaves 0 for the 'unknown' token.
    Path to the pretrained word vectors
    Enable verbose output.

Attributes
----------
    an instance of :class:`pytorch_widedeep.utils.fastai_transforms.Vocab`
    Array with the pretrained embeddings
    List with Lists of str containing the tokenized texts

Examples
---------

```python

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
```


**Methods:**


### .__init__
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/preprocessing/text_preprocessor.py/#L65)
```python
.__init__(
   text_col: str, max_vocab: int = 30000, min_freq: int = 5, maxlen: int = 80,
   pad_first: bool = True, pad_idx: int = 1, word_vectors_path: Optional[str] = None,
   verbose: int = 1
)
```


### .inverse_transform
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/preprocessing/text_preprocessor.py/#L125)
```python
.inverse_transform(
   padded_seq: np.ndarray
)
```

---
Returns the original text plus the added 'special' tokens

### .fit_transform
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/preprocessing/text_preprocessor.py/#L121)
```python
.fit_transform(
   df: pd.DataFrame
)
```

---
Combines ``fit`` and ``transform``

### .transform
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/preprocessing/text_preprocessor.py/#L102)
```python
.transform(
   df: pd.DataFrame
)
```

---
Returns the padded, `numericalised` sequences

### .fit
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/preprocessing/text_preprocessor.py/#L87)
```python
.fit(
   df: pd.DataFrame
)
```

---
Builds the vocabulary
