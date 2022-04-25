#


## FTTransformer
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/models/tabular/transformers/ft_transformer.py/#L13)
```python 
FTTransformer(
   column_idx: Dict[str, int], cat_embed_input: Optional[List[Tuple[str,
   int]]] = None, cat_embed_dropout: float = 0.1, use_cat_bias: bool = False,
   cat_embed_activation: Optional[str] = None, full_embed_dropout: bool = False,
   shared_embed: bool = False, add_shared_embed: bool = False,
   frac_shared_embed: float = 0.25, continuous_cols: Optional[List[str]] = None,
   cont_norm_layer: str = None, cont_embed_dropout: float = 0.1,
   use_cont_bias: bool = True, cont_embed_activation: Optional[str] = None,
   input_dim: int = 64, kv_compression_factor: float = 0.5, kv_sharing: bool = False,
   use_qkv_bias: bool = False, n_heads: int = 8, n_blocks: int = 4,
   attn_dropout: float = 0.2, ff_dropout: float = 0.1,
   transformer_activation: str = 'reglu', ff_factor: float = 1.33,
   mlp_hidden_dims: Optional[List[int]] = None, mlp_activation: str = 'relu',
   mlp_dropout: float = 0.1, mlp_batchnorm: bool = False,
   mlp_batchnorm_last: bool = False, mlp_linear_first: bool = True
)
```


---
Defines a `FTTransformer model <https://arxiv.org/abs/2106.11959>`_ that
can be used as the ``deeptabular`` component of a Wide & Deep model or
independently by itself.


Parameters
----------
column_idx: Dict
Dict containing the index of the columns that will be passed through
the model. Required to slice the tensors. e.g.
{'education': 0, 'relationship': 1, 'workclass': 2, ...}
---
    each categorical component e.g. [(education, 11), ...]
    Categorical embeddings dropout
    Boolean indicating if bias will be used for the categorical embeddings
    `'relu'`, `'leaky_relu'` and `'gelu'` are supported.
    If ``full_embed_dropout = True``, ``cat_embed_dropout`` is ignored.
    at the time.
    See :obj:`pytorch_widedeep.models.transformers._layers.SharedEmbeddings`
    column.
    List with the name of the numeric (aka continuous) columns
    are: 'layernorm', 'batchnorm' or None.
    Continuous embeddings dropout
    Boolean indicating if bias will be used for the continuous embeddings
    any. `'tanh'`, `'relu'`, `'leaky_relu'` and `'gelu'` are supported.
    the categorical and/or continuous columns.
    where :math:`s` is the input sequence length.
    Complexity <https://arxiv.org/abs/2006.04768>`_ for details
    Number of attention heads per FTTransformer block
    projection layers
    Number of FTTransformer blocks
    Dropout that will be applied to the Linear-Attention layers
    Dropout that will be applied to the FeedForward network
    `'leaky_relu'`, `'gelu'`, `'geglu'` and `'reglu'` are supported
    in the paper.
    FTTransformer block will be used
    `'gelu'` are supported
    Dropout that will be applied to the final MLP
    dense layers
    last of the dense layers
    LIN -> ACT]``

Attributes
----------
    This is the module that processes the categorical and continuous columns
    Sequence of FTTransformer blocks
    MLP component in the model
    neccesary to build the ``WideDeep`` class

Example
--------

```python

>>> from pytorch_widedeep.models import FTTransformer
>>> X_tab = torch.cat((torch.empty(5, 4).random_(4), torch.rand(5, 1)), axis=1)
>>> colnames = ['a', 'b', 'c', 'd', 'e']
>>> cat_embed_input = [(u,i) for u,i in zip(colnames[:4], [4]*4)]
>>> continuous_cols = ['e']
>>> column_idx = {k:v for v,k in enumerate(colnames)}
>>> model = FTTransformer(column_idx=column_idx, cat_embed_input=cat_embed_input, continuous_cols=continuous_cols)
>>> out = model(X_tab)
```


**Methods:**


### .__init__
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/models/tabular/transformers/ft_transformer.py/#L145)
```python
.__init__(
   column_idx: Dict[str, int], cat_embed_input: Optional[List[Tuple[str,
   int]]] = None, cat_embed_dropout: float = 0.1, use_cat_bias: bool = False,
   cat_embed_activation: Optional[str] = None, full_embed_dropout: bool = False,
   shared_embed: bool = False, add_shared_embed: bool = False,
   frac_shared_embed: float = 0.25, continuous_cols: Optional[List[str]] = None,
   cont_norm_layer: str = None, cont_embed_dropout: float = 0.1,
   use_cont_bias: bool = True, cont_embed_activation: Optional[str] = None,
   input_dim: int = 64, kv_compression_factor: float = 0.5, kv_sharing: bool = False,
   use_qkv_bias: bool = False, n_heads: int = 8, n_blocks: int = 4,
   attn_dropout: float = 0.2, ff_dropout: float = 0.1,
   transformer_activation: str = 'reglu', ff_factor: float = 1.33,
   mlp_hidden_dims: Optional[List[int]] = None, mlp_activation: str = 'relu',
   mlp_dropout: float = 0.1, mlp_batchnorm: bool = False,
   mlp_batchnorm_last: bool = False, mlp_linear_first: bool = True
)
```


### .forward
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/models/tabular/transformers/ft_transformer.py/#L267)
```python
.forward(
   X: Tensor
)
```


### .attention_weights
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/models/tabular/transformers/ft_transformer.py/#L279)
```python
.attention_weights()
```

---
List with the attention weights per block

The shape of the attention weights is:

:math:`(N, H, F, k)`

where *N* is the batch size, *H* is the number of attention heads, *F*
is the number of features/columns and *k* is the reduced sequence
length or dimension, i.e. :math:`k = int(kv_
{compression \space factor} \times s)`
