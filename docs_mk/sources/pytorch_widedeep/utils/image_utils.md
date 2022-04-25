#


## SimplePreprocessor
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/utils/image_utils.py/#L81)
```python 
SimplePreprocessor(
   width: int, height: int, inter = cv2.INTER_AREA
)
```


---
Class to resize an image to a certain width and height

Parameters
----------
width: int
output width
---
    output height
        formatting error.


**Methods:**


### .__init__
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/utils/image_utils.py/#L98)
```python
.__init__(
   width: int, height: int, inter = cv2.INTER_AREA
)
```


### .preprocess
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/utils/image_utils.py/#L103)
```python
.preprocess(
   image: np.ndarray
)
```

---
Returns the resized input image

Parameters
----------
image: np.ndarray
Input image to be resized
