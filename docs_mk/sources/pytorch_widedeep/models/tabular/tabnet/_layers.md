#


## TabNetEncoder
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/models/tabular/tabnet/_layers.py/#L224)
```python 
TabNetEncoder(
   input_dim: int, n_steps: int = 3, step_dim: int = 8, attn_dim: int = 8,
   dropout: float = 0.0, n_glu_step_dependent: int = 2, n_glu_shared: int = 2,
   ghost_bn: bool = True, virtual_batch_size: int = 128, momentum: float = 0.02,
   gamma: float = 1.3, epsilon: float = 1e-15, mask_type: str = 'sparsemax'
)
```




**Methods:**


### .__init__
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/models/tabular/tabnet/_layers.py/#L225)
```python
.__init__(
   input_dim: int, n_steps: int = 3, step_dim: int = 8, attn_dim: int = 8,
   dropout: float = 0.0, n_glu_step_dependent: int = 2, n_glu_shared: int = 2,
   ghost_bn: bool = True, virtual_batch_size: int = 128, momentum: float = 0.02,
   gamma: float = 1.3, epsilon: float = 1e-15, mask_type: str = 'sparsemax'
)
```


### .forward
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/models/tabular/tabnet/_layers.py/#L297)
```python
.forward(
   X: Tensor
)
```


### .forward_masks
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/models/tabular/tabnet/_layers.py/#L334)
```python
.forward_masks(
   X: Tensor
)
```


----


### initialize_non_glu
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/models/tabular/tabnet/_layers.py/#L22)
```python
.initialize_non_glu(
   module, input_dim: int, output_dim: int
)
```


----


### initialize_glu
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/models/tabular/tabnet/_layers.py/#L28)
```python
.initialize_glu(
   module, input_dim: int, output_dim: int
)
```

