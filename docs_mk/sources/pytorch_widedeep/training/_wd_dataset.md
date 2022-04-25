#


## WideDeepDataset
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/training/_wd_dataset.py/#L14)
```python 
WideDeepDataset(
   X_wide: Optional[np.ndarray] = None, X_tab: Optional[np.ndarray] = None,
   X_text: Optional[np.ndarray] = None, X_img: Optional[np.ndarray] = None,
   target: Optional[np.ndarray] = None, transforms: Optional[Any] = None,
   with_lds: bool = False, lds_kernel: Literal['gaussian', 'triang',
   'laplace'] = 'gaussian', lds_ks: int = 5, lds_sigma: float = 2,
   lds_granularity: int = 100, lds_reweight: bool = False,
   lds_y_max: Optional[float] = None, lds_y_min: Optional[float] = None,
   is_training: bool = True
)
```


---
Defines the Dataset object to load WideDeep data to the model

Parameters
----------
X_wide: np.ndarray
wide input
---
    deeptabular input
    deeptext input
    deepimage input
    target array
    torchvision Compose object. See models/_multiple_transforms.py
    the dataset
    choice of kernel for Label Distribution Smoothing
    LDS kernel window size
    standard deviation of ['gaussian','laplace'] kernel for LDS
    number of bins in histogram used in LDS to count occurence of sample values
    option to reweight bin frequency counts in LDS
    option to restrict LDS bins by upper label limit
    option to restrict LDS bins by lower label limit


**Methods:**


### .__init__
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/training/_wd_dataset.py/#L51)
```python
.__init__(
   X_wide: Optional[np.ndarray] = None, X_tab: Optional[np.ndarray] = None,
   X_text: Optional[np.ndarray] = None, X_img: Optional[np.ndarray] = None,
   target: Optional[np.ndarray] = None, transforms: Optional[Any] = None,
   with_lds: bool = False, lds_kernel: Literal['gaussian', 'triang',
   'laplace'] = 'gaussian', lds_ks: int = 5, lds_sigma: float = 2,
   lds_granularity: int = 100, lds_reweight: bool = False,
   lds_y_max: Optional[float] = None, lds_y_min: Optional[float] = None,
   is_training: bool = True
)
```


### ._compute_lds_weights
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/training/_wd_dataset.py/#L121)
```python
._compute_lds_weights(
   lds_y_min: Optional[float], lds_y_max: Optional[float], granularity: int,
   reweight: bool, kernel: Literal['gaussian', 'triang', 'laplace'], ks: int,
   sigma: float
)
```

---
Assign weight to each sample by following procedure:
1.      creating histogram from label values with nuber of bins = granularity
2[opt]. reweighting label frequencies by sqrt
3[opt]. smoothing label frequencies by convolution of kernel function window with frequencies list
4.      inverting values by n_samples / (n_classes * np.bincount(y)), see:
https://scikit-learn.org/stable/modules/generated/sklearn.utils.class_weight.compute_sample_weight.html
5.      assigning weight to each sample from closest bin value

### .__len__
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/training/_wd_dataset.py/#L197)
```python
.__len__()
```


### ._prepare_images
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/training/_wd_dataset.py/#L169)
```python
._prepare_images(
   idx
)
```


### .__getitem__
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/training/_wd_dataset.py/#L102)
```python
.__getitem__(
   idx: int
)
```

