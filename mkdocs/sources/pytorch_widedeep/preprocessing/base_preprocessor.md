#


## BasePreprocessor
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/preprocessing/base_preprocessor.py/#L10)
```python 
BasePreprocessor(
   *args
)
```


---
Base Class of All Preprocessors.


**Methods:**


### .transform
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/preprocessing/base_preprocessor.py/#L19)
```python
.transform(
   df: pd.DataFrame
)
```


### .__init__
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/preprocessing/base_preprocessor.py/#L13)
```python
.__init__(
   *args
)
```


### .fit
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/preprocessing/base_preprocessor.py/#L16)
```python
.fit(
   df: pd.DataFrame
)
```


### .fit_transform
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/preprocessing/base_preprocessor.py/#L22)
```python
.fit_transform(
   df: pd.DataFrame
)
```


----


### check_is_fitted
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/preprocessing/base_preprocessor.py/#L26)
```python
.check_is_fitted(
   estimator: BasePreprocessor, attributes: List[str] = None, all_or_any: str = 'all',
   condition: bool = True
)
```

---
Checks if an estimator is fitted

Parameters
----------
estimator: ``BasePreprocessor``,
An object of type ``BasePreprocessor``
---
    List of strings with the attributes to check for
    whether all or any of the attributes in the list must be present
    the estimator to be considered as fitted
