#


## TabFastFormer
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/models/tabular/transformers/tab_fastformer.py/#L13)
```python 
TabFastFormer(
   column_idx: Dict[str, int], cat_embed_input: Optional[List[Tuple[str,
   int]]] = None, cat_embed_dropout: float = 0.1, use_cat_bias: bool = False,
   cat_embed_activation: Optional[str] = None, full_embed_dropout: bool = False,
   shared_embed: bool = False, add_shared_embed: bool = False,
   frac_shared_embed: float = 0.25, continuous_cols: Optional[List[str]] = None,
   cont_norm_layer: str = None, cont_embed_dropout: float = 0.1,
   use_cont_bias: bool = True, cont_embed_activation: Optional[str] = None,
   input_dim: int = 32, n_heads: int = 8, use_bias: bool = False, n_blocks: int = 4,
   attn_dropout: float = 0.1, ff_dropout: float = 0.2, share_qv_weights: bool = False,
   share_weights: bool = False, transformer_activation: str = 'relu',
   mlp_hidden_dims: Optional[List[int]] = None, mlp_activation: str = 'relu',
   mlp_dropout: float = 0.1, mlp_batchnorm: bool = False,
   mlp_batchnorm_last: bool = False, mlp_linear_first: bool = True
)
```


---
Defines an adaptation of a `FastFormer model
<https://arxiv.org/abs/2108.09084>`_ that can be used as the
``deeptabular`` component of a Wide & Deep model or independently by
itself.

Parameters
----------
column_idx: Dict
Dict containing the index of the columns that will be passed through
the ``TabMlp`` model. Required to slice the tensors. e.g. {'education':
0, 'relationship': 1, 'workclass': 2, ...}
---
    embedding dimension. e.g. [(education, 11, 32), ...]
    Categorical embeddings dropout
    Boolean indicating if bias will be used for the categorical embeddings
    Activation function for the categorical embeddings
    If ``full_embed_dropout = True``, ``cat_embed_dropout`` is ignored.
    at the time.
    See :obj:`pytorch_widedeep.models.transformers._layers.SharedEmbeddings`
    column.
    List with the name of the numeric (aka continuous) columns
    are: 'layernorm', 'batchnorm' or None.
    Continuous embeddings dropout
    Boolean indicating if bias will be used for the continuous embeddings
    `'gelu'` are supported.
    embeddings used to encode the categorical and/or continuous columns
    Number of attention heads per FastFormer block
    projection layers
    Number of FastFormer blocks
    Dropout that will be applied to the Additive Attention layers
    Dropout that will be applied to the FeedForward network
    the query transformation parameters will be shared
    the parameters across different Fastformer layers can also be shared
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
    Sequence of FasFormer blocks.
    MLP component in the model
    neccesary to build the ``WideDeep`` class

Example
--------

```python

>>> from pytorch_widedeep.models import TabFastFormer
>>> X_tab = torch.cat((torch.empty(5, 4).random_(4), torch.rand(5, 1)), axis=1)
>>> colnames = ['a', 'b', 'c', 'd', 'e']
>>> cat_embed_input = [(u,i) for u,i in zip(colnames[:4], [4]*4)]
>>> continuous_cols = ['e']
>>> column_idx = {k:v for v,k in enumerate(colnames)}
>>> model = TabFastFormer(column_idx=column_idx, cat_embed_input=cat_embed_input, continuous_cols=continuous_cols)
>>> out = model(X_tab)
```


**Methods:**


### .__init__
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/models/tabular/transformers/tab_fastformer.py/#L135)
```python
.__init__(
   column_idx: Dict[str, int], cat_embed_input: Optional[List[Tuple[str,
   int]]] = None, cat_embed_dropout: float = 0.1, use_cat_bias: bool = False,
   cat_embed_activation: Optional[str] = None, full_embed_dropout: bool = False,
   shared_embed: bool = False, add_shared_embed: bool = False,
   frac_shared_embed: float = 0.25, continuous_cols: Optional[List[str]] = None,
   cont_norm_layer: str = None, cont_embed_dropout: float = 0.1,
   use_cont_bias: bool = True, cont_embed_activation: Optional[str] = None,
   input_dim: int = 32, n_heads: int = 8, use_bias: bool = False, n_blocks: int = 4,
   attn_dropout: float = 0.1, ff_dropout: float = 0.2, share_qv_weights: bool = False,
   share_weights: bool = False, transformer_activation: str = 'relu',
   mlp_hidden_dims: Optional[List[int]] = None, mlp_activation: str = 'relu',
   mlp_dropout: float = 0.1, mlp_batchnorm: bool = False,
   mlp_batchnorm_last: bool = False, mlp_linear_first: bool = True
)
```


### .attention_weights
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/models/tabular/transformers/tab_fastformer.py/#L277)
```python
.attention_weights()
```

---
List with the attention weights. Each element of the list is a
tuple where the first and second elements are :math:`\alpha`
and :math:`\beta` attention weights in the paper.

The shape of the attention weights is:

:math:`(N, H, F)`

where *N* is the batch size, *H* is the number of attention heads
and *F* is the number of features/columns in the dataset

### .forward
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/models/tabular/transformers/tab_fastformer.py/#L267)
```python
.forward(
   X: Tensor
)
```

