#


## TabPerceiver
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/models/tabular/transformers/tab_perceiver.py/#L15)
```python 
TabPerceiver(
   column_idx: Dict[str, int], cat_embed_input: Optional[List[Tuple[str,
   int]]] = None, cat_embed_dropout: float = 0.1, use_cat_bias: bool = False,
   cat_embed_activation: Optional[str] = None, full_embed_dropout: bool = False,
   shared_embed: bool = False, add_shared_embed: bool = False,
   frac_shared_embed: float = 0.25, continuous_cols: Optional[List[str]] = None,
   cont_norm_layer: str = None, cont_embed_dropout: float = 0.1,
   use_cont_bias: bool = True, cont_embed_activation: Optional[str] = None,
   input_dim: int = 32, n_cross_attns: int = 1, n_cross_attn_heads: int = 4,
   n_latents: int = 16, latent_dim: int = 128, n_latent_heads: int = 4,
   n_latent_blocks: int = 4, n_perceiver_blocks: int = 4, share_weights: bool = False,
   attn_dropout: float = 0.1, ff_dropout: float = 0.1,
   transformer_activation: str = 'geglu', mlp_hidden_dims: Optional[List[int]] = None,
   mlp_activation: str = 'relu', mlp_dropout: float = 0.1, mlp_batchnorm: bool = False,
   mlp_batchnorm_last: bool = False, mlp_linear_first: bool = True
)
```


---
Defines an adaptation of a `Perceiver model
<https://arxiv.org/abs/2103.03206>`_ that can be used as the
``deeptabular`` component of a Wide & Deep model or independently by
itself.

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
    used to encode the categorical and/or continuous columns.
    cases for tabular data
    Number of attention heads for the cross attention component
    the transformer quadratic bottleneck
    Latent dimension.
    Number of attention heads per Latent Transformer
    per Latent Transformer
    Transformer]
    blocks
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
    ModuleDict with the Perceiver blocks
    Latents that will be used for prediction
    MLP component in the model
    neccesary to build the ``WideDeep`` class

Example
--------

```python

>>> from pytorch_widedeep.models import TabPerceiver
>>> X_tab = torch.cat((torch.empty(5, 4).random_(4), torch.rand(5, 1)), axis=1)
>>> colnames = ['a', 'b', 'c', 'd', 'e']
>>> cat_embed_input = [(u,i) for u,i in zip(colnames[:4], [4]*4)]
>>> continuous_cols = ['e']
>>> column_idx = {k:v for v,k in enumerate(colnames)}
>>> model = TabPerceiver(column_idx=column_idx, cat_embed_input=cat_embed_input,
... continuous_cols=continuous_cols, n_latents=2, latent_dim=16,
... n_perceiver_blocks=2)
>>> out = model(X_tab)
```


**Methods:**


### .__init__
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/models/tabular/transformers/tab_perceiver.py/#L157)
```python
.__init__(
   column_idx: Dict[str, int], cat_embed_input: Optional[List[Tuple[str,
   int]]] = None, cat_embed_dropout: float = 0.1, use_cat_bias: bool = False,
   cat_embed_activation: Optional[str] = None, full_embed_dropout: bool = False,
   shared_embed: bool = False, add_shared_embed: bool = False,
   frac_shared_embed: float = 0.25, continuous_cols: Optional[List[str]] = None,
   cont_norm_layer: str = None, cont_embed_dropout: float = 0.1,
   use_cont_bias: bool = True, cont_embed_activation: Optional[str] = None,
   input_dim: int = 32, n_cross_attns: int = 1, n_cross_attn_heads: int = 4,
   n_latents: int = 16, latent_dim: int = 128, n_latent_heads: int = 4,
   n_latent_blocks: int = 4, n_perceiver_blocks: int = 4, share_weights: bool = False,
   attn_dropout: float = 0.1, ff_dropout: float = 0.1,
   transformer_activation: str = 'geglu', mlp_hidden_dims: Optional[List[int]] = None,
   mlp_activation: str = 'relu', mlp_dropout: float = 0.1, mlp_batchnorm: bool = False,
   mlp_batchnorm_last: bool = False, mlp_linear_first: bool = True
)
```


### ._build_perceiver_block
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/models/tabular/transformers/tab_perceiver.py/#L326)
```python
._build_perceiver_block()
```


### ._extract_attn_weights
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/models/tabular/transformers/tab_perceiver.py/#L365)
```python
._extract_attn_weights(
   cross_attns, latent_transformer
)
```


### .forward
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/models/tabular/transformers/tab_perceiver.py/#L267)
```python
.forward(
   X: Tensor
)
```


### .attention_weights
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/models/tabular/transformers/tab_perceiver.py/#L288)
```python
.attention_weights()
```

---
List with the attention weights. If the weights are not shared
between perceiver blocks each element of the list will be a list
itself containing the Cross Attention and Latent Transformer
attention weights respectively

The shape of the attention weights is:

- Cross Attention: :math:`(N, C, L, F)`
- Latent Attention: :math:`(N, T, L, L)`

---
WHere *N* is the batch size, *C* is the number of Cross Attention
heads, *L* is the number of Latents, *F* is the number of
features/columns in the dataset and *T* is the number of Latent
Attention heads
