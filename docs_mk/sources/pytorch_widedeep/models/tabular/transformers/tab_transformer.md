#


## TabTransformer
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/models/tabular/transformers/tab_transformer.py/#L14)
```python 
TabTransformer(
   column_idx: Dict[str, int], cat_embed_input: Optional[List[Tuple[str,
   int]]] = None, cat_embed_dropout: float = 0.1, use_cat_bias: bool = False,
   cat_embed_activation: Optional[str] = None, full_embed_dropout: bool = False,
   shared_embed: bool = False, add_shared_embed: bool = False,
   frac_shared_embed: float = 0.25, continuous_cols: Optional[List[str]] = None,
   cont_norm_layer: str = None, embed_continuous: bool = False,
   cont_embed_dropout: float = 0.1, use_cont_bias: bool = True,
   cont_embed_activation: Optional[str] = None, input_dim: int = 32, n_heads: int = 8,
   use_qkv_bias: bool = False, n_blocks: int = 4, attn_dropout: float = 0.2,
   ff_dropout: float = 0.1, transformer_activation: str = 'gelu',
   mlp_hidden_dims: Optional[List[int]] = None, mlp_activation: str = 'relu',
   mlp_dropout: float = 0.1, mlp_batchnorm: bool = False,
   mlp_batchnorm_last: bool = False, mlp_linear_first: bool = True
)
```


---
Defines a `TabTransformer model <https://arxiv.org/abs/2012.06678>`_ that
can be used as the ``deeptabular`` component of a Wide & Deep model or
independently by itself.

Note that this is an enhanced adaptation of the model described in the
paper, containing a series of additional features.

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
    (i.e. passed each through a linear layer with or without activation)
    Continuous embeddings dropout
    Boolean indicating if bias will be used for the continuous embeddings
    any. `'tanh'`, `'relu'`, `'leaky_relu'` and `'gelu'` are supported.
    embeddings used to encode the categorical and/or continuous columns
    Number of attention heads per Transformer block
    projection layers.
    Number of Transformer blocks
    Dropout that will be applied to the Multi-Head Attention layers
    Dropout that will be applied to the FeedForward network
    `'leaky_relu'`, `'gelu'`, `'geglu'` and `'reglu'` are supported
    2*l]`` where ``l`` is the MLP's input dimension
    `'gelu'` are supported
    Dropout that will be applied to the final MLP
    dense layers
    last of the dense layers
    LIN -> ACT]``

Attributes
----------
    This is the module that processes the categorical and continuous columns
    Sequence of Transformer blocks
    MLP component in the model
    neccesary to build the ``WideDeep`` class

Example
--------

```python

>>> from pytorch_widedeep.models import TabTransformer
>>> X_tab = torch.cat((torch.empty(5, 4).random_(4), torch.rand(5, 1)), axis=1)
>>> colnames = ['a', 'b', 'c', 'd', 'e']
>>> cat_embed_input = [(u,i) for u,i in zip(colnames[:4], [4]*4)]
>>> continuous_cols = ['e']
>>> column_idx = {k:v for v,k in enumerate(colnames)}
>>> model = TabTransformer(column_idx=column_idx, cat_embed_input=cat_embed_input, continuous_cols=continuous_cols)
>>> out = model(X_tab)
```


**Methods:**


### .__init__
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/models/tabular/transformers/tab_transformer.py/#L135)
```python
.__init__(
   column_idx: Dict[str, int], cat_embed_input: Optional[List[Tuple[str,
   int]]] = None, cat_embed_dropout: float = 0.1, use_cat_bias: bool = False,
   cat_embed_activation: Optional[str] = None, full_embed_dropout: bool = False,
   shared_embed: bool = False, add_shared_embed: bool = False,
   frac_shared_embed: float = 0.25, continuous_cols: Optional[List[str]] = None,
   cont_norm_layer: str = None, embed_continuous: bool = False,
   cont_embed_dropout: float = 0.1, use_cont_bias: bool = True,
   cont_embed_activation: Optional[str] = None, input_dim: int = 32, n_heads: int = 8,
   use_qkv_bias: bool = False, n_blocks: int = 4, attn_dropout: float = 0.2,
   ff_dropout: float = 0.1, transformer_activation: str = 'gelu',
   mlp_hidden_dims: Optional[List[int]] = None, mlp_activation: str = 'relu',
   mlp_dropout: float = 0.1, mlp_batchnorm: bool = False,
   mlp_batchnorm_last: bool = False, mlp_linear_first: bool = True
)
```


### .forward
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/models/tabular/transformers/tab_transformer.py/#L247)
```python
.forward(
   X: Tensor
)
```


### .attention_weights
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/models/tabular/transformers/tab_transformer.py/#L273)
```python
.attention_weights()
```

---
List with the attention weights per block

The shape of the attention weights is:

:math:`(N, H, F, F)`

Where *N* is the batch size, *H* is the number of attention heads
and *F* is the number of features/columns in the dataset

### ._compute_attn_output_dim
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/models/tabular/transformers/tab_transformer.py/#L285)
```python
._compute_attn_output_dim()
```

