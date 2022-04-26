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


**Args**

* **column_idx** (Dict) : Dict containing the index of the columns that will be passed through
    the model. Required to slice the tensors. e.g.
    {'education': 0, 'relationship': 1, 'workclass': 2, ...}
* **cat_embed_input** (List, Optional, default = None) : List of Tuples with the column name and number of unique values for
    each categorical component e.g. [(education, 11), ...]
* **cat_embed_dropout** (float, default = 0.1) : Categorical embeddings dropout
* **use_cat_bias** (bool, default = False) : Boolean indicating if bias will be used for the categorical embeddings
* **cat_embed_activation** (Optional, str, default = None) : Activation function for the categorical embeddings, if any. `'tanh'`,
    `'relu'`, `'leaky_relu'` and `'gelu'` are supported.
* **full_embed_dropout** (bool, default = False) : Boolean indicating if an entire embedding (i.e. the representation of
    one column) will be dropped in the batch. See:
    :obj:`pytorch_widedeep.models.transformers._layers.FullEmbeddingDropout`.
    If ``full_embed_dropout = True``, ``cat_embed_dropout`` is ignored.
* **shared_embed** (bool, default = False) : The idea behind ``shared_embed`` is described in the Appendix A in the
    `TabTransformer paper <https://arxiv.org/abs/2012.06678>`_: `'The
    goal of having column embedding is to enable the model to distinguish
    the classes in one column from those in the other columns'`. In other
    words, the idea is to let the model learn which column is embedded
    at the time.
* **add_shared_embed** (bool, default = False) : The two embedding sharing strategies are: 1) add the shared embeddings
    to the column embeddings or 2) to replace the first
    ``frac_shared_embed`` with the shared embeddings.
    See :obj:`pytorch_widedeep.models.transformers._layers.SharedEmbeddings`
* **frac_shared_embed** (float, default = 0.25) : The fraction of embeddings that will be shared (if ``add_shared_embed
    = False``) by all the different categories for one particular
    column.
* **continuous_cols** (List, Optional, default = None) : List with the name of the numeric (aka continuous) columns
* **cont_norm_layer** (str, default =  "batchnorm") : Type of normalization layer applied to the continuous features. Options
    are: 'layernorm', 'batchnorm' or None.
* **cont_embed_dropout** (float, default = 0.1) : Continuous embeddings dropout
* **use_cont_bias** (bool, default = True) : Boolean indicating if bias will be used for the continuous embeddings
* **cont_embed_activation** (str, default = None) : Activation function to be applied to the continuous embeddings, if
    any. `'tanh'`, `'relu'`, `'leaky_relu'` and `'gelu'` are supported.
* **input_dim** (int, default = 32) : The so-called *dimension of the model*. Is the number of embeddings
    used to encode the categorical and/or continuous columns.
* **n_cross_attns** (int, default = 1) : Number of times each perceiver block will cross attend to the input
    data (i.e. number of cross attention components per perceiver block).
    This should normally be 1. However, in the paper they describe some
    architectures (normally computer vision-related problems) where the
    Perceiver attends multiple times to the input array. Therefore, maybe
    multiple cross attention to the input array is also useful in some
    cases for tabular data
* **n_cross_attn_heads** (int, default = 4) : Number of attention heads for the cross attention component
* **n_latents** (int, default = 16) : Number of latents. This is the *N* parameter in the paper. As
    indicated in the paper, this number should be significantly lower
    than *M* (the number of columns in the dataset). Setting *N* closer
    to *M* defies the main purpose of the Perceiver, which is to overcome
    the transformer quadratic bottleneck
* **latent_dim** (int, default = 128) : Latent dimension.
* **n_latent_heads** (int, default = 4) : Number of attention heads per Latent Transformer
* **n_latent_blocks** (int, default = 4) : Number of transformer encoder blocks (normalised MHA + normalised FF)
    per Latent Transformer
* **n_perceiver_blocks** (int, default = 4) : Number of Perceiver blocks defined as [Cross Attention + Latent
    Transformer]
* **share_weights** (Boolean, default = False) : Boolean indicating if the weights will be shared between Perceiver
    blocks
* **attn_dropout** (float, default = 0.2) : Dropout that will be applied to the Multi-Head Attention layers
* **ff_dropout** (float, default = 0.1) : Dropout that will be applied to the FeedForward network
* **transformer_activation** (str, default = "gelu") : Transformer Encoder activation function. `'tanh'`, `'relu'`,
    `'leaky_relu'`, `'gelu'`, `'geglu'` and `'reglu'` are supported
* **mlp_hidden_dims** (List, Optional, default = None) : MLP hidden dimensions. If not provided it will default to ``[l, 4*l,
    2*l]`` where ``l`` is the MLP's input dimension
* **mlp_activation** (str, default = "relu") : MLP activation function. `'tanh'`, `'relu'`, `'leaky_relu'` and
    `'gelu'` are supported
* **mlp_dropout** (float, default = 0.1) : Dropout that will be applied to the final MLP
* **mlp_batchnorm** (bool, default = False) : Boolean indicating whether or not to apply batch normalization to the
    dense layers
* **mlp_batchnorm_last** (bool, default = False) : Boolean indicating whether or not to apply batch normalization to the
    last of the dense layers
* **mlp_linear_first** (bool, default = False) : Boolean indicating whether the order of the operations in the dense
    layer. If ``True: [LIN -> ACT -> BN -> DP]``. If ``False: [BN -> DP ->
    LIN -> ACT]``


**Attributes**

* **cat_and_cont_embed** (nn.Module) : This is the module that processes the categorical and continuous columns
* **perceiver_blks** (nn.ModuleDict) : ModuleDict with the Perceiver blocks
* **latents** (nn.Parameter) : Latents that will be used for prediction
* **perceiver_mlp** (nn.Module) : MLP component in the model
* **output_dim** (int) : The output dimension of the model. This is a required attribute
    neccesary to build the ``WideDeep`` class


**Example**


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


### .forward
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/models/tabular/transformers/tab_perceiver.py/#L227)
```python
.forward(
   X: Tensor
)
```


### .attention_weights
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/models/tabular/transformers/tab_perceiver.py/#L248)
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
