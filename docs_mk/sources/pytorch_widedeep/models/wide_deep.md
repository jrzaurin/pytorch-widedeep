#


## WideDeep
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/models/wide_deep.py/#L27)
```python 
WideDeep(
   wide: Optional[nn.Module] = None, deeptabular: Optional[nn.Module] = None,
   deeptext: Optional[nn.Module] = None, deepimage: Optional[nn.Module] = None,
   deephead: Optional[nn.Module] = None, head_hidden_dims: Optional[List[int]] = None,
   head_activation: str = 'relu', head_dropout: float = 0.1,
   head_batchnorm: bool = False, head_batchnorm_last: bool = False,
   head_linear_first: bool = False, enforce_positive: bool = False,
   enforce_positive_activation: str = 'softplus', pred_dim: int = 1,
   with_fds: bool = False, **fds_config
)
```


---
Main collector class that combines all ``wide``, ``deeptabular``
``deeptext`` and ``deepimage`` models.

There are two options to combine these models that correspond to the
two main architectures that ``pytorch-widedeep`` can build.

- Directly connecting the output of the model components to an ouput neuron(s).

- Adding a `Fully-Connected Head` (FC-Head) on top of the deep models.
  This FC-Head will combine the output form the ``deeptabular``, ``deeptext`` and
  ``deepimage`` and will be then connected to the output neuron(s).

---
Parameters
----------
    captured via crossed-columns.
    package.
    package.
    documenation of the package.
    List with the sizes of the dense layers in the head e.g: [128, 64]
    `'tanh'`, `'relu'`, `'leaky_relu'` and `'gelu'` are supported
    Dropout of the dense layers in the head
    the dense layers that form the `'rnn_mlp'`
    last of the dense layers in the head
    LIN -> ACT]``
    previous fc-head parameters will be ignored
    predictions are bounded in between 0 and inf
    output. `'softplus'` or `'relu'` are supported.
    of classes for multiclass classification.
    for details.
    ``FeatureDistributionSmoothing`` layer

Examples
--------


```python

>>> embed_input = [(u, i, j) for u, i, j in zip(["a", "b", "c"][:4], [4] * 3, [8] * 3)]
>>> column_idx = {k: v for v, k in enumerate(["a", "b", "c"])}
>>> wide = Wide(10, 1)
>>> deeptabular = TabResnet(blocks_dims=[8, 4], column_idx=column_idx, cat_embed_input=embed_input)
>>> deeptext = BasicRNN(vocab_size=10, embed_dim=4, padding_idx=0)
>>> deepimage = Vision()
>>> model = WideDeep(wide=wide, deeptabular=deeptabular, deeptext=deeptext, deepimage=deepimage)

```

    example :class:`pytorch_widedeep.models.tab_mlp.TabMlp`


**Methods:**


### ._add_pred_layer
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/models/wide_deep.py/#L247)
```python
._add_pred_layer()
```


### .__init__
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/models/wide_deep.py/#L122)
```python
.__init__(
   wide: Optional[nn.Module] = None, deeptabular: Optional[nn.Module] = None,
   deeptext: Optional[nn.Module] = None, deepimage: Optional[nn.Module] = None,
   deephead: Optional[nn.Module] = None, head_hidden_dims: Optional[List[int]] = None,
   head_activation: str = 'relu', head_dropout: float = 0.1,
   head_batchnorm: bool = False, head_batchnorm_last: bool = False,
   head_linear_first: bool = False, enforce_positive: bool = False,
   enforce_positive_activation: str = 'softplus', pred_dim: int = 1,
   with_fds: bool = False, **fds_config
)
```


### ._forward_deep_with_fds
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/models/wide_deep.py/#L321)
```python
._forward_deep_with_fds(
   X: Dict[str, Tensor], y: Optional[Tensor] = None, epoch: Optional[int] = None
)
```


### .forward
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/models/wide_deep.py/#L195)
```python
.forward(
   X: Dict[str, Tensor], y: Optional[Tensor] = None, epoch: Optional[int] = None
)
```


### ._forward_wide
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/models/wide_deep.py/#L269)
```python
._forward_wide(
   X
)
```


### ._forward_deephead
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/models/wide_deep.py/#L278)
```python
._forward_deephead(
   X, wide_out
)
```


### ._build_deephead
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/models/wide_deep.py/#L216)
```python
._build_deephead(
   head_hidden_dims, head_activation, head_dropout, head_batchnorm,
   head_batchnorm_last, head_linear_first
)
```


### ._forward_deep
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/models/wide_deep.py/#L302)
```python
._forward_deep(
   X, wide_out
)
```


### ._check_inputs
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/models/wide_deep.py/#L338)
```python
._check_inputs(
   wide, deeptabular, deeptext, deepimage, deephead, head_hidden_dims, pred_dim,
   with_fds
)
```

