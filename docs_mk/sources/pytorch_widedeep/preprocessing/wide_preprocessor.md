#


## WidePreprocessor
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/preprocessing/wide_preprocessor.py/#L11)
```python 
WidePreprocessor(
   wide_cols: List[str], crossed_cols: List[Tuple[str, str]] = None
)
```


---
Preprocessor to prepare the wide input dataset

This Preprocessor prepares the data for the wide, linear component.
This linear model is implemented via an Embedding layer that is
connected to the output neuron. ``WidePreprocessor`` numerically
encodes all the unique values of all categorical columns ``wide_cols +
crossed_cols``. See the Example below.

Parameters
----------
wide_cols: List
List of strings with the name of the columns that will label
encoded and passed through the ``wide`` component
---
    the constituent features are all 1, and 0 otherwise".

Attributes
----------
    List with the names of all columns that will be label encoded
    column value` and the values are the corresponding mapped integer.
    Dimension of the wide model (i.e. dim of the linear layer)

Example
-------

```python

>>> from pytorch_widedeep.preprocessing import WidePreprocessor
>>> df = pd.DataFrame({'color': ['r', 'b', 'g'], 'size': ['s', 'n', 'l']})
>>> wide_cols = ['color']
>>> crossed_cols = [('color', 'size')]
>>> wide_preprocessor = WidePreprocessor(wide_cols=wide_cols, crossed_cols=crossed_cols)
>>> X_wide = wide_preprocessor.fit_transform(df)
>>> X_wide
array([[1, 4],
       [2, 5],
       [3, 6]])
>>> wide_preprocessor.encoding_dict
{'color_r': 1, 'color_b': 2, 'color_g': 3, 'color_size_r-s': 4, 'color_size_b-n': 5, 'color_size_g-l': 6}
>>> wide_preprocessor.inverse_transform(X_wide)
  color color_size
0     r        r-s
1     b        b-n
2     g        g-l
```


**Methods:**


### .__init__
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/preprocessing/wide_preprocessor.py/#L63)
```python
.__init__(
   wide_cols: List[str], crossed_cols: List[Tuple[str, str]] = None
)
```


### ._make_global_feature_list
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/preprocessing/wide_preprocessor.py/#L113)
```python
._make_global_feature_list(
   df: pd.DataFrame
)
```


### .inverse_transform
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/preprocessing/wide_preprocessor.py/#L98)
```python
.inverse_transform(
   encoded: np.ndarray
)
```

---
Takes as input the output from the ``transform`` method and it will
return the original values.

### .fit_transform
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/preprocessing/wide_preprocessor.py/#L109)
```python
.fit_transform(
   df: pd.DataFrame
)
```

---
Combines ``fit`` and ``transform``

### ._make_column_feature_list
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/preprocessing/wide_preprocessor.py/#L119)
```python
._make_column_feature_list(
   s: pd.Series
)
```


### ._cross_cols
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/preprocessing/wide_preprocessor.py/#L122)
```python
._cross_cols(
   df: pd.DataFrame
)
```


### .transform
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/preprocessing/wide_preprocessor.py/#L85)
```python
.transform(
   df: pd.DataFrame
)
```

---
Returns the processed dataframe

### ._prepare_wide
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/preprocessing/wide_preprocessor.py/#L133)
```python
._prepare_wide(
   df: pd.DataFrame
)
```


### .fit
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/preprocessing/wide_preprocessor.py/#L71)
```python
.fit(
   df: pd.DataFrame
)
```

---
Fits the Preprocessor and creates required attributes
