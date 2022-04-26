#


## ContextAttentionEncoder
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/models/text/_encoders.py/#L13)
```python 
ContextAttentionEncoder(
   rnn: nn.Module, input_dim: int, attn_dropout: float, attn_concatenate: bool,
   with_addnorm: bool, sum_along_seq: bool
)
```




**Methods:**


### .__init__
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/models/text/_encoders.py/#L14)
```python
.__init__(
   rnn: nn.Module, input_dim: int, attn_dropout: float, attn_concatenate: bool,
   with_addnorm: bool, sum_along_seq: bool
)
```


### .forward
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/models/text/_encoders.py/#L35)
```python
.forward(
   X: Tensor, h: Tensor, c: Tensor
)
```


### ._process_rnn_outputs
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/models/text/_encoders.py/#L51)
```python
._process_rnn_outputs(
   output: Tensor, hidden: Tensor
)
```

