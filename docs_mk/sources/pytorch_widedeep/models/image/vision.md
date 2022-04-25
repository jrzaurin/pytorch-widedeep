#


## Vision
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/models/image/vision.py/#L24)
```python 
Vision(
   pretrained_model_name: Optional[str] = None, n_trainable: Optional[int] = None,
   trainable_params: Optional[List[str]] = None, channel_sizes: List[int] = [64, 128,
   256, 512], kernel_sizes: Union[int, List[int]] = [7, 3, 3, 3],
   strides: Union[int, List[int]] = [2, 1, 1, 1],
   head_hidden_dims: Optional[List[int]] = None, head_activation: str = 'relu',
   head_dropout: Union[float, List[float]] = 0.1, head_batchnorm: bool = False,
   head_batchnorm_last: bool = False, head_linear_first: bool = False
)
```


---
Defines a standard image classifier/regressor using a pretrained
network or a sequence of convolution layers that can be used as the
``deepimage`` component of a Wide & Deep model or independently by
itself.

Parameters
----------
pretrained_model_name: Optional, str, default = None
Name of the pretrained model. Should be a variant of the following
architectures: `'resnet`', `'shufflenet`', `'resnext`',
`'wide_resnet`', `'regnet`', `'densenet`', `'mobilenetv3`',
`'mobilenetv2`', `'mnasnet`', `'efficientnet`' and `'squeezenet`'. if
`pretrained_model_name = None` a basic, fully trainable CNN will be
used.
---
    ``trainable_params`` is not None this parameter will be ignored
    used.
    to use a pretrained model
    (channel_sizes) - 1`.
    (channel_sizes) - 1`.
    List with the number of neurons per dense layer in the head. e.g: [64,32]
    `'tanh'`, `'relu'`, `'leaky_relu'` and `'gelu'` are supported
    float indicating the dropout between the dense layers.
    to the dense layers
    to the last of the dense layers
    LIN -> ACT]``

Attributes
----------
    The pretrained model or Standard CNN plus the optional head
    neccesary to build the ``WideDeep`` class

Example
--------

```python

>>> from pytorch_widedeep.models import Vision
>>> X_img = torch.rand((2,3,224,224))
>>> model = Vision(channel_sizes=[64, 128], kernel_sizes = [3, 3], strides=[1, 1], head_hidden_dims=[32, 8])
>>> out = model(X_img)
```


**Methods:**


### ._get_features
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/models/image/vision.py/#L175)
```python
._get_features()
```


### .__init__
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/models/image/vision.py/#L98)
```python
.__init__(
   pretrained_model_name: Optional[str] = None, n_trainable: Optional[int] = None,
   trainable_params: Optional[List[str]] = None, channel_sizes: List[int] = [64, 128,
   256, 512], kernel_sizes: Union[int, List[int]] = [7, 3, 3, 3],
   strides: Union[int, List[int]] = [2, 1, 1, 1],
   head_hidden_dims: Optional[List[int]] = None, head_activation: str = 'relu',
   head_dropout: Union[float, List[float]] = 0.1, head_batchnorm: bool = False,
   head_batchnorm_last: bool = False, head_linear_first: bool = False
)
```


### ._freeze
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/models/image/vision.py/#L217)
```python
._freeze(
   features
)
```


### .forward
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/models/image/vision.py/#L161)
```python
.forward(
   X: Tensor
)
```


### ._basic_cnn
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/models/image/vision.py/#L188)
```python
._basic_cnn()
```


### .get_backbone_output_dim
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/models/image/vision.py/#L235)
```python
.get_backbone_output_dim(
   features
)
```

