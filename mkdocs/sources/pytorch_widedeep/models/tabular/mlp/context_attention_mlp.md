#


## ContextAttentionMLP
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/models/tabular/mlp/context_attention_mlp.py/#L12)
```python 
ContextAttentionMLP(
   column_idx: Dict[str, int], cat_embed_input: Optional[List[Tuple[str,
   int]]] = None, cat_embed_dropout: float = 0.1, use_cat_bias: bool = False,
   cat_embed_activation: Optional[str] = None, full_embed_dropout: bool = False,
   shared_embed: bool = False, add_shared_embed: bool = False,
   frac_shared_embed: float = 0.25, continuous_cols: Optional[List[str]] = None,
   cont_norm_layer: str = None, cont_embed_dropout: float = 0.1,
   use_cont_bias: bool = True, cont_embed_activation: Optional[str] = None,
   input_dim: int = 32, attn_dropout: float = 0.2, with_addnorm: bool = False,
   attn_activation: str = 'leaky_relu', n_blocks: int = 3
)
```


---
Defines a ``ContextAttentionMLP`` model that can be used as the
``deeptabular`` component of a Wide & Deep model or independently by
itself.

This class combines embedding representations of the categorical features
with numerical (aka continuous) features that are also embedded. These
are then passed through a series of attention blocks. Each attention
block is comprised by a ``ContextAttentionEncoder``.
See :obj:`pytorch_widedeep.models.tabular.mlp._attention_layers` for
details


Parameters
----------
column_idx: Dict
Dict containing the index of the columns that will be passed through
the model. Required to slice the tensors. e.g.
{'education': 0, 'relationship': 1, 'workclass': 2, ...}
---
    categorical columns. e.g. [(education, 11), ...]
    Categorical embeddings dropout
    Boolean indicating if bias will be used for the categorical embeddings
    `'relu'`, `'leaky_relu'` and `'gelu'` are supported.
    If ``full_embed_dropout = True``, ``cat_embed_dropout`` is ignored.
    the model learn which column is embedded at the time.
    See :obj:`pytorch_widedeep.models.embeddings_layers.SharedEmbeddings`
    column.
    List with the name of the numeric (aka continuous) columns
    are: 'layernorm', 'batchnorm' or None.
    Continuous embeddings dropout
    Boolean indicating if bias will be used for the continuous embeddings
    any. `'tanh'`, `'relu'`, `'leaky_relu'` and `'gelu'` are supported.
    used to encode the categorical and/or continuous columns
    Dropout for each attention block
    attention blocks
    and `'gelu'` are supported.
    Number of attention blocks

Attributes
----------
    This is the module that processes the categorical and continuous columns
    Sequence of attention encoders.
    neccesary to build the ``WideDeep`` class

Example
--------

```python

>>> from pytorch_widedeep.models import ContextAttentionMLP
>>> X_tab = torch.cat((torch.empty(5, 4).random_(4), torch.rand(5, 1)), axis=1)
>>> colnames = ['a', 'b', 'c', 'd', 'e']
>>> cat_embed_input = [(u,i,j) for u,i,j in zip(colnames[:4], [4]*4, [8]*4)]
>>> column_idx = {k:v for v,k in enumerate(colnames)}
>>> model = ContextAttentionMLP(column_idx=column_idx, cat_embed_input=cat_embed_input, continuous_cols = ['e'])
>>> out = model(X_tab)
```


**Methods:**


### .__init__
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/models/tabular/mlp/context_attention_mlp.py/#L107)
```python
.__init__(
   column_idx: Dict[str, int], cat_embed_input: Optional[List[Tuple[str,
   int]]] = None, cat_embed_dropout: float = 0.1, use_cat_bias: bool = False,
   cat_embed_activation: Optional[str] = None, full_embed_dropout: bool = False,
   shared_embed: bool = False, add_shared_embed: bool = False,
   frac_shared_embed: float = 0.25, continuous_cols: Optional[List[str]] = None,
   cont_norm_layer: str = None, cont_embed_dropout: float = 0.1,
   use_cont_bias: bool = True, cont_embed_activation: Optional[str] = None,
   input_dim: int = 32, attn_dropout: float = 0.2, with_addnorm: bool = False,
   attn_activation: str = 'leaky_relu', n_blocks: int = 3
)
```


### .attention_weights
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/models/tabular/mlp/context_attention_mlp.py/#L187)
```python
.attention_weights()
```

---
List with the attention weights per block

The shape of the attention weights is:

:math:`(N, F)`

Where *N* is the batch size and *F* is the number of features/columns
in the dataset

### .forward
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/models/tabular/mlp/context_attention_mlp.py/#L177)
```python
.forward(
   X: Tensor
)
```

