#


## QueryKeySelfAttention
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/models/tabular/mlp/_attention_layers.py/#L32)
```python 
QueryKeySelfAttention(
   input_dim: int, dropout: float, use_bias: bool, n_heads: int
)
```


---
Attention mechanism inspired by the well known multi-head attention. Here,
rather than learning a value projection matrix that will be multiplied by
the attention weights, we multiply such weights directly by the input
tensor.

The rationale behind this implementation comes, among other
considerations, from the fact that Transformer based models tend to
heavily overfit tabular. Therefore, by reducing the number of trainable
parameters and multiply directly by the incoming tensor we help
mitigating such overfitting


**Methods:**


### .__init__
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/models/tabular/mlp/_attention_layers.py/#L45)
```python
.__init__(
   input_dim: int, dropout: float, use_bias: bool, n_heads: int
)
```


### .forward
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/models/tabular/mlp/_attention_layers.py/#L61)
```python
.forward(
   X: Tensor
)
```

