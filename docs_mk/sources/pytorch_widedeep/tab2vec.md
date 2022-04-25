#


## Tab2Vec
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/tab2vec.py/#L15)
```python 
Tab2Vec(
   model: Union[WideDeep, BaseBayesianModel], tab_preprocessor: TabPreprocessor,
   return_dataframe: bool = False, verbose: bool = False
)
```


---
Class to transform an input dataframe into vectorized form.

This class will take an input dataframe in the form of the dataframe used
for training, and it will turn it into a vectorised form based on the
processing applied by the model to the categorical and continuous
columns.

.. note:: Currently this class is only implemented for the deeptabular
component or the Bayesian model. Therefore, if the input dataframe has
a text column or a column with the path to images, these will be
ignored. We will be adding these functionalities in future versions

---
Parameters
----------
    ``WideDeep`` ``BaseBayesianModel`` model. Must be trained.
    ``TabPreprocessor`` object. Must be fitted.
    pandas dataframe(s)

Attributes
----------
    Torch module with the categorical and continuous encoding process

Examples
--------

```python

>>> from random import choices
>>> import numpy as np
>>> import pandas as pd
>>> from pytorch_widedeep import Tab2Vec
>>> from pytorch_widedeep.models import TabMlp, WideDeep
>>> from pytorch_widedeep.preprocessing import TabPreprocessor
>>>
>>> colnames = list(string.ascii_lowercase)[:4]
>>> cat_col1_vals = ["a", "b", "c"]
>>> cat_col2_vals = ["d", "e", "f"]
>>>
>>> # Create the toy input dataframe and a toy dataframe to be vectorised
>>> cat_inp = [np.array(choices(c, k=5)) for c in [cat_col1_vals, cat_col2_vals]]
>>> cont_inp = [np.round(np.random.rand(5), 2) for _ in range(2)]
>>> df_inp = pd.DataFrame(np.vstack(cat_inp + cont_inp).transpose(), columns=colnames)
>>> cat_t2v = [np.array(choices(c, k=5)) for c in [cat_col1_vals, cat_col2_vals]]
>>> cont_t2v = [np.round(np.random.rand(5), 2) for _ in range(2)]
>>> df_t2v = pd.DataFrame(np.vstack(cat_t2v + cont_t2v).transpose(), columns=colnames)
>>>
>>> # fit the TabPreprocessor
>>> embed_cols = [("a", 2), ("b", 4)]
>>> cont_cols = ["c", "d"]
>>> tab_preprocessor = TabPreprocessor(cat_embed_cols=embed_cols, continuous_cols=cont_cols)
>>> X_tab = tab_preprocessor.fit_transform(df_inp)
>>>
>>> # define the model (and let's assume we train it)
>>> tabmlp = TabMlp(
... column_idx=tab_preprocessor.column_idx,
... cat_embed_input=tab_preprocessor.cat_embed_input,
... continuous_cols=tab_preprocessor.continuous_cols,
... mlp_hidden_dims=[8, 4])
>>> model = WideDeep(deeptabular=tabmlp)
>>> # ...train the model...
>>>
>>> # vectorise the dataframe
>>> t2v = Tab2Vec(model, tab_preprocessor)
>>> X_vec = t2v.transform(df_t2v)
```


**Methods:**


### .__init__
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/tab2vec.py/#L85)
```python
.__init__(
   model: Union[WideDeep, BaseBayesianModel], tab_preprocessor: TabPreprocessor,
   return_dataframe: bool = False, verbose: bool = False
)
```


### ._new_colnames_with_attn
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/tab2vec.py/#L209)
```python
._new_colnames_with_attn()
```


### ._new_colnames
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/tab2vec.py/#L202)
```python
._new_colnames()
```


### ._new_colnames_without_attn
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/tab2vec.py/#L235)
```python
._new_colnames_without_attn()
```


### .fit_transform
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/tab2vec.py/#L191)
```python
.fit_transform(
   df: pd.DataFrame, target_col: Optional[str] = None
)
```

---
Combines ``fit`` and ``transform``

### .transform
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/tab2vec.py/#L138)
```python
.transform(
   df: pd.DataFrame, target_col: Optional[str] = None
)
```

---
Transforms the input dataframe into vectorized form. If a target
column name is passed the target values will be returned separately
in their corresponding type (np.ndarray or pd.DataFrame)

Parameters
----------
df: pd.DataFrame
DataFrame to be vectorised, i.e. the categorical and continuous
columns will be encoded based on the processing applied within
the model
---
    predictors will be returned

### .fit
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/tab2vec.py/#L132)
```python
.fit(
   df: pd.DataFrame, target_col: Optional[str] = None
)
```

---
Empty method. Returns the object itself. Is only included for
consistency in case ``Tab2Vec`` is used as part of a Pipeline
