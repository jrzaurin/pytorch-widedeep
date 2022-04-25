#


## FDSLayer
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/models/fds_layer.py/#L12)
```python 
FDSLayer(
   feature_dim: int, granularity: int = 100, y_max: Optional[float] = None,
   y_min: Optional[float] = None, start_update: int = 0, start_smooth: int = 2,
   kernel: Literal['gaussian', 'triang', 'laplace'] = 'gaussian', ks: int = 5,
   sigma: float = 2, momentum: Optional[float] = 0.9, clip_min: Optional[float] = None,
   clip_max: Optional[float] = None
)
```




**Methods:**


### .__init__
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/models/fds_layer.py/#L13)
```python
.__init__(
   feature_dim: int, granularity: int = 100, y_max: Optional[float] = None,
   y_min: Optional[float] = None, start_update: int = 0, start_smooth: int = 2,
   kernel: Literal['gaussian', 'triang', 'laplace'] = 'gaussian', ks: int = 5,
   sigma: float = 2, momentum: Optional[float] = 0.9, clip_min: Optional[float] = None,
   clip_max: Optional[float] = None
)
```

---
Feature Distribution Smoothing layer. Layer keeps track of last epoch mean
and variance for each feature. The feautres are bucket-ed into bins based on
their target/label value. Target/label values are binned using histogram with
same width bins, their number is based on granularity parameter and start/end
edge on y_max/y_min values. Mean and variance are smoothed using convolution
with kernel function(gaussian by default). Output of the layer are features
values adjusted to their smoothed mean and variance. The layer is turned on
only during training, off during prediction/evaluation.

Adjusted code from `<https://github.com/YyzHarry/imbalanced-regression>`
For more infomation about please read the paper) :

`Yang, Y., Zha, K., Chen, Y. C., Wang, H., & Katabi, D. (2021).
Delving into Deep Imbalanced Regression. arXiv preprint arXiv:2102.09554.`

Parameters
----------
feature_dim: int,
input dimension size, i.e. output size of previous layer
---
    values per label
    option to restrict the histogram bins by upper label limit
    option to restrict the histogram bins by lower label limit
    epoch after which FDS layer starts to update its statistics
    epoch after which FDS layer starts to smooth feautture distributions
    choice of kernel for Feature Distribution Smoothing
    LDS kernel window size
    standard deviation of ['gaussian','laplace'] kernel for LDS
    factor parameter for running mean and variance
    estimation may not be stable
    original value = 10, see note for clip_min

### .forward
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/models/fds_layer.py/#L96)
```python
.forward(
   features, labels, epoch
)
```


### ._smooth
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/models/fds_layer.py/#L198)
```python
._smooth(
   features, labels, epoch
)
```


### ._register_buffers
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/models/fds_layer.py/#L241)
```python
._register_buffers()
```


### .reset
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/models/fds_layer.py/#L103)
```python
.reset()
```


### .update_last_epoch_stats
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/models/fds_layer.py/#L112)
```python
.update_last_epoch_stats(
   epoch
)
```


### ._calibrate_mean_var
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/models/fds_layer.py/#L214)
```python
._calibrate_mean_var(
   features, left_bin_edge_ind
)
```


### .update_running_stats
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/models/fds_layer.py/#L153)
```python
.update_running_stats(
   features, labels, epoch
)
```

