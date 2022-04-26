#


## TabPreprocessor
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/preprocessing/tab_preprocessor.py/#L39)
```python 
TabPreprocessor(
   cat_embed_cols: Union[List[str], List[Tuple[str, int]]] = None,
   continuous_cols: List[str] = None, scale: bool = True, auto_embed_dim: bool = True,
   embedding_rule: Literal['google', 'fastai_old', 'fastai_new'] = 'fastai_new',
   default_embed_dim: int = 16, already_standard: List[str] = None,
   with_attention: bool = False, with_cls_token: bool = False,
   shared_embed: bool = False, verbose: int = 1
)
```


---
Preprocessor to prepare the ``deeptabular`` component input dataset

Parameters
----------
cat_embed_cols: List, default = None
List containing the name of the categorical columns that will be
represented by embeddings (e.g.['education', 'relationship', ...]) or
a Tuple with the name and the embedding dimension (e.g.:[
('education',32),('relationship',16), ...])
---
    List with the name of the continuous cols
    Param alias: ``scale_cont_cols``

    scaled/standarised.
    below.
    - `'google'` -- :math:`min(600, round(n_{cat}^{0.24}))`

    ``False``.
    Param alias: ``for_transformer``

    being passed to the final MLP (if present).
    :obj:`pytorch_widedeep.models.transformers._layers.SharedEmbeddings`.
verbose: int, default = 1

Attributes
----------
    is not generated during the ``fit`` process
    see :class:`pytorch_widedeep.utils.dense_utils.LabelEncder`
    ('relationship', 6, 8), ...].
    List of the columns that will be standarized
    an instance of :class:`sklearn.preprocessing.StandardScaler`
    This is neccesary to slice tensors

Examples
--------

```python

>>> from pytorch_widedeep.preprocessing import TabPreprocessor
>>> df = pd.DataFrame({'color': ['r', 'b', 'g'], 'size': ['s', 'n', 'l'], 'age': [25, 40, 55]})
>>> cat_embed_cols = [('color',5), ('size',5)]
>>> cont_cols = ['age']
>>> deep_preprocessor = TabPreprocessor(cat_embed_cols=cat_embed_cols, continuous_cols=cont_cols)
>>> X_tab = deep_preprocessor.fit_transform(df)
>>> deep_preprocessor.embed_dim
{'color': 5, 'size': 5}
>>> deep_preprocessor.column_idx
{'color': 0, 'size': 1, 'age': 2}
```


**Methods:**


### ._prepare_embed
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/preprocessing/tab_preprocessor.py/#L260)
```python
._prepare_embed(
   df: pd.DataFrame
)
```


### .__init__
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/preprocessing/tab_preprocessor.py/#L147)
```python
.__init__(
   cat_embed_cols: Union[List[str], List[Tuple[str, int]]] = None,
   continuous_cols: List[str] = None, scale: bool = True, auto_embed_dim: bool = True,
   embedding_rule: Literal['google', 'fastai_old', 'fastai_new'] = 'fastai_new',
   default_embed_dim: int = 16, already_standard: List[str] = None,
   with_attention: bool = False, with_cls_token: bool = False,
   shared_embed: bool = False, verbose: int = 1
)
```


### .inverse_transform
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/preprocessing/tab_preprocessor.py/#L225)
```python
.inverse_transform(
   encoded: np.ndarray
)
```

---
Takes as input the output from the ``transform`` method and it will
return the original values.

Parameters
----------
encoded: np.ndarray
array with the output of the ``transform`` method

### .fit
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/preprocessing/tab_preprocessor.py/#L178)
```python
.fit(
   df: pd.DataFrame
)
```

---
Fits the Preprocessor and creates required attributes

### ._check_inputs
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/preprocessing/tab_preprocessor.py/#L294)
```python
._check_inputs()
```


### .fit_transform
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/preprocessing/tab_preprocessor.py/#L256)
```python
.fit_transform(
   df: pd.DataFrame
)
```

---
Combines ``fit`` and ``transform``

### ._prepare_continuous
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/preprocessing/tab_preprocessor.py/#L284)
```python
._prepare_continuous(
   df: pd.DataFrame
)
```


### .transform
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/preprocessing/tab_preprocessor.py/#L204)
```python
.transform(
   df: pd.DataFrame
)
```

---
Returns the processed ``dataframe`` as a np.ndarray

----


### embed_sz_rule
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/preprocessing/tab_preprocessor.py/#L16)
```python
.embed_sz_rule(
   n_cat: int, embedding_rule: Literal['google', 'fastai_old',
   'fastai_new'] = 'fastai_new'
)
```

---
Rule of thumb to pick embedding size corresponding to ``n_cat``. Default rule is taken
from recent fastai's Tabular API. The function also includes previously used rule by fastai
and rule included in the Google's Tensorflow documentation

Parameters
----------
n_cat: int
number of unique categorical values in a feature
---
    rule of thumb to be used for embedding vector size
