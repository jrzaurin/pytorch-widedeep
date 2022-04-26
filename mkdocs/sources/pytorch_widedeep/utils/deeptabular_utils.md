#


## LabelEncoder
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/utils/deeptabular_utils.py/#L19)
```python 
LabelEncoder(
   columns_to_encode: Optional[List[str]] = None, with_attention: bool = False,
   shared_embed: bool = False
)
```


---
Label Encode categorical values for multiple columns at once

.. note:: LabelEncoder reserves 0 for `unseen` new categories. This is convenient
when defining the embedding layers, since we can just set padding idx to 0.

---
Parameters
----------
    encoded.
    Aliased as ``for_transformer``.
    :obj:`pytorch_widedeep.models.transformers._layers.SharedEmbeddings`.

Attributes
-----------
    `{'colname1': {'cat1': 1, 'cat2': 2, ...}, 'colname2': {'cat1': 1, 'cat2': 2, ...}, ...}`

    `{'colname1': {1: 'cat1', 2: 'cat2', ...}, 'colname2': {1: 'cat1', 2: 'cat2', ...}, ...}`


**Methods:**


### .__init__
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/utils/deeptabular_utils.py/#L59)
```python
.__init__(
   columns_to_encode: Optional[List[str]] = None, with_attention: bool = False,
   shared_embed: bool = False
)
```


### .inverse_transform
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/utils/deeptabular_utils.py/#L154)
```python
.inverse_transform(
   df: pd.DataFrame
)
```

---
Returns the original categories

Examples
--------


```python

>>> from pytorch_widedeep.utils import LabelEncoder
>>> df = pd.DataFrame({'col1': [1,2,3], 'col2': ['me', 'you', 'him']})
>>> columns_to_encode = ['col2']
>>> encoder = LabelEncoder(columns_to_encode)
>>> df_enc = encoder.fit_transform(df)
>>> encoder.inverse_transform(df_enc)
   col1 col2
0     1   me
1     2  you
2     3  him
```

### .fit
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/utils/deeptabular_utils.py/#L72)
```python
.fit(
   df: pd.DataFrame
)
```

---
Creates encoding attributes

### .fit_transform
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/utils/deeptabular_utils.py/#L133)
```python
.fit_transform(
   df: pd.DataFrame
)
```

---
Combines ``fit`` and ``transform``

Examples
--------


```python

>>> from pytorch_widedeep.utils import LabelEncoder
>>> df = pd.DataFrame({'col1': [1,2,3], 'col2': ['me', 'you', 'him']})
>>> columns_to_encode = ['col2']
>>> encoder = LabelEncoder(columns_to_encode)
>>> encoder.fit_transform(df)
   col1  col2
0     1     1
1     2     2
2     3     3
>>> encoder.encoding_dict
{'col2': {'me': 1, 'you': 2, 'him': 3}}
```

### .transform
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/utils/deeptabular_utils.py/#L112)
```python
.transform(
   df: pd.DataFrame
)
```

---
Label Encoded the categories in ``columns_to_encode``

----


### find_bin
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/utils/deeptabular_utils.py/#L177)
```python
.find_bin(
   bin_edges: Union[np.ndarray, Tensor], values: Union[np.ndarray, Tensor],
   ret_value: bool = True
)
```

---
Returns histograms left bin edge value or array indices from monotonically
increasing array of bin edges for each value in values.
If ret_value

Parameters
----------
bin_edges: Union[np.ndarray, Tensor]
monotonically increasing array of bin edges
---
    values for which we want corresponding bins
    if True, return bin values else indices

Returns
-------
    left bin edges

----


### _laplace
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/utils/deeptabular_utils.py/#L222)
```python
._laplace(
   x
)
```


----


### get_kernel_window
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/utils/deeptabular_utils.py/#L226)
```python
.get_kernel_window(
   kernel: Literal['gaussian', 'triang', 'laplace'] = 'gaussian', ks: int = 5,
   sigma: Union[int, float] = 2
)
```

---
Procedure to prepare window of values from symetrical kernel function for smoothing of the distribution in
Label and Feature Distribution Smoothing (LDS & FDS).

Parameters
----------
kernel: Literal['gaussian', 'triang', 'laplace'] = 'gaussian'
choice of kernel for label distribution smoothing
---
    kernel size, i.e. count of samples in symmetric window
    standard deviation of ['gaussian','laplace'] kernel

Returns
-------
    list with values from the chosen kernel function
