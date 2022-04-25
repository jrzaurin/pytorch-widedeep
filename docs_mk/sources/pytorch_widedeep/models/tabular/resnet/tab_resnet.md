#


## TabResnet
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/models/tabular/resnet/tab_resnet.py/#L9)
```python 
TabResnet(
   column_idx: Dict[str, int], cat_embed_input: Optional[List[Tuple[str, int,
   int]]] = None, cat_embed_dropout: float = 0.1, use_cat_bias: bool = False,
   cat_embed_activation: Optional[str] = None,
   continuous_cols: Optional[List[str]] = None, cont_norm_layer: str = 'batchnorm',
   embed_continuous: bool = False, cont_embed_dim: int = 32,
   cont_embed_dropout: float = 0.1, use_cont_bias: bool = True,
   cont_embed_activation: Optional[str] = None, blocks_dims: List[int] = [200, 100,
   100], blocks_dropout: float = 0.1, simplify_blocks: bool = False,
   mlp_hidden_dims: Optional[List[int]] = None, mlp_activation: str = 'relu',
   mlp_dropout: float = 0.1, mlp_batchnorm: bool = False,
   mlp_batchnorm_last: bool = False, mlp_linear_first: bool = False
)
```


---
Defines a ``TabResnet`` model that can be used as the ``deeptabular``
component of a Wide & Deep model or independently by itself.

This class combines embedding representations of the categorical features
with numerical (aka continuous) features, embedded or not. These are then
passed through a series of Resnet blocks. See
:obj:`pytorch_widedeep.models.tab_resnet._layers` for details on the
structure of each block.

Parameters
----------
column_idx: Dict
Dict containing the index of the columns that will be passed through
the model. Required to slice the tensors. e.g.
{'education': 0, 'relationship': 1, 'workclass': 2, ...}
---
    embedding dimension. e.g. [(education, 11, 32), ...].
    Categorical embeddings dropout
    Boolean indicating if bias will be used for the categorical embeddings
    `'tanh'`, `'relu'`, `'leaky_relu'` and `'gelu'` are supported
    List with the name of the numeric (aka continuous) columns
    are: 'layernorm', 'batchnorm' or None.
    (i.e. passed each through a linear layer with or without activation)
    Size of the continuous embeddings
    Continuous embeddings dropout
    Boolean indicating if bias will be used for the continuous embeddings
    `'tanh'`, `'relu'`, `'leaky_relu'` and `'gelu'` are supported
    details on the structure of each block.
blocks_dropout: float, default =  0.1
   Block's `"internal"` dropout.
    (``X -> [ [LIN1, BN1, ACT1] -> [LIN2, BN2]  + X ]``).
    optional.
    `'tanh'`, `'relu'`, `'leaky_relu'` and `'gelu'` are supported
    float with the dropout between the dense layers of the MLP.
    to the dense layers
    to the last of the dense layers
    LIN -> ACT]``

Attributes
----------
    This is the module that processes the categorical and continuous columns
    embeddings and the continuous columns
    embeddings and the continuous columns -- if present --.
    neccesary to build the ``WideDeep`` class

Example
--------

```python

>>> from pytorch_widedeep.models import TabResnet
>>> X_deep = torch.cat((torch.empty(5, 4).random_(4), torch.rand(5, 1)), axis=1)
>>> colnames = ['a', 'b', 'c', 'd', 'e']
>>> cat_embed_input = [(u,i,j) for u,i,j in zip(colnames[:4], [4]*4, [8]*4)]
>>> column_idx = {k:v for v,k in enumerate(colnames)}
>>> model = TabResnet(blocks_dims=[16,4], column_idx=column_idx, cat_embed_input=cat_embed_input,
... continuous_cols = ['e'])
>>> out = model(X_deep)
```


**Methods:**


### .__init__
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/models/tabular/resnet/tab_resnet.py/#L114)
```python
.__init__(
   column_idx: Dict[str, int], cat_embed_input: Optional[List[Tuple[str, int,
   int]]] = None, cat_embed_dropout: float = 0.1, use_cat_bias: bool = False,
   cat_embed_activation: Optional[str] = None,
   continuous_cols: Optional[List[str]] = None, cont_norm_layer: str = 'batchnorm',
   embed_continuous: bool = False, cont_embed_dim: int = 32,
   cont_embed_dropout: float = 0.1, use_cont_bias: bool = True,
   cont_embed_activation: Optional[str] = None, blocks_dims: List[int] = [200, 100,
   100], blocks_dropout: float = 0.1, simplify_blocks: bool = False,
   mlp_hidden_dims: Optional[List[int]] = None, mlp_activation: str = 'relu',
   mlp_dropout: float = 0.1, mlp_batchnorm: bool = False,
   mlp_batchnorm_last: bool = False, mlp_linear_first: bool = False
)
```


### .forward
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/models/tabular/resnet/tab_resnet.py/#L194)
```python
.forward(
   X: Tensor
)
```

