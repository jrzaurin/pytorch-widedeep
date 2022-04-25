#


## MLP
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/models/tabular/mlp/_layers.py/#L52)
```python 
MLP(
   d_hidden: List[int], activation: str, dropout: Optional[Union[float,
   List[float]]], batchnorm: bool, batchnorm_last: bool, linear_first: bool
)
```




**Methods:**


### .__init__
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/models/tabular/mlp/_layers.py/#L53)
```python
.__init__(
   d_hidden: List[int], activation: str, dropout: Optional[Union[float,
   List[float]]], batchnorm: bool, batchnorm_last: bool, linear_first: bool
)
```


### .forward
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/models/tabular/mlp/_layers.py/#L83)
```python
.forward(
   X: Tensor
)
```


----


### dense_layer
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/models/tabular/mlp/_layers.py/#L7)
```python
.dense_layer(
   inp: int, out: int, activation: str, p: float, bn: bool, linear_first: bool
)
```

