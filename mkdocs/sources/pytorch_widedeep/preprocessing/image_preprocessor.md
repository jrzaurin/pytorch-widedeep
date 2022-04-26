#


## ImagePreprocessor
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/preprocessing/image_preprocessor.py/#L16)
```python 
ImagePreprocessor(
   img_col: str, img_path: str, width: int = 224, height: int = 224, verbose: int = 1
)
```


---
Preprocessor to prepare the ``deepimage`` input dataset.

The Preprocessing consists simply on resizing according to their
aspect ratio

Parameters
----------
img_col: str
name of the column with the images filenames
---
    path to the dicrectory where the images are stored
    width of the resulting processed image.
    width of the resulting processed image.
    Enable verbose output.

Attributes
----------
    an instance of :class:`pytorch_widedeep.utils.image_utils.AspectAwarePreprocessor`
    an instance of :class:`pytorch_widedeep.utils.image_utils.SimplePreprocessor`
    mean and std for the R, G and B channels

Examples
--------

```python

>>>
>>> from pytorch_widedeep.preprocessing import ImagePreprocessor
>>>
>>> path_to_image1 = 'tests/test_data_utils/images/galaxy1.png'
>>> path_to_image2 = 'tests/test_data_utils/images/galaxy2.png'
>>>
>>> df_train = pd.DataFrame({'images_column': [path_to_image1]})
>>> df_test = pd.DataFrame({'images_column': [path_to_image2]})
>>> img_preprocessor = ImagePreprocessor(img_col='images_column', img_path='.', verbose=0)
>>> resized_images = img_preprocessor.fit_transform(df_train)
>>> new_resized_images = img_preprocessor.transform(df_train)

```
    instantiates the resizing functions.


**Methods:**


### .__init__
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/preprocessing/image_preprocessor.py/#L66)
```python
.__init__(
   img_col: str, img_path: str, width: int = 224, height: int = 224, verbose: int = 1
)
```


### .inverse_transform
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/preprocessing/image_preprocessor.py/#L156)
```python
.inverse_transform(
   transformed_image
)
```


### .fit
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/preprocessing/image_preprocessor.py/#L82)
```python
.fit(
   df: pd.DataFrame
)
```

---
Instantiates the Preprocessors
:obj:`AspectAwarePreprocessor`` and :obj:`SimplePreprocessor` for image
resizing.

See
:class:`pytorch_widedeep.utils.image_utils.AspectAwarePreprocessor`
and :class:`pytorch_widedeep.utils.image_utils.SimplePreprocessor`.

### .fit_transform
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/preprocessing/image_preprocessor.py/#L152)
```python
.fit_transform(
   df: pd.DataFrame
)
```

---
Combines ``fit`` and ``transform``

### .transform
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/preprocessing/image_preprocessor.py/#L97)
```python
.transform(
   df: pd.DataFrame
)
```

---
Resizes the images to the input height and width.
