#


## BayesianTabMlp
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/bayesian_models/tabular/bayesian_mlp/bayesian_tab_mlp.py/#L16)
```python 
BayesianTabMlp(
   column_idx: Dict[str, int], cat_embed_input: Optional[List[Tuple[str, int,
   int]]] = None, cat_embed_dropout: float = 0.1,
   cat_embed_activation: Optional[str] = None,
   continuous_cols: Optional[List[str]] = None, embed_continuous: bool = False,
   cont_embed_dim: int = 32, cont_embed_dropout: float = 0.1,
   cont_embed_activation: Optional[str] = None, use_cont_bias: bool = True,
   cont_norm_layer: str = 'batchnorm', mlp_hidden_dims: List[int] = [200, 100],
   mlp_activation: str = 'leaky_relu', prior_sigma_1: float = 1,
   prior_sigma_2: float = 0.002, prior_pi: float = 0.8, posterior_mu_init: float = 0.0,
   posterior_rho_init: float = -7.0, pred_dim = 1
)
```


---
Defines a ``BayesianTabMlp`` model.

This class combines embedding representations of the categorical features
with numerical (aka continuous) features, embedded or not. These are then
passed through a series of dense layers (i.e. a MLP).

Parameters
----------
column_idx: Dict
Dict containing the index of the columns that will be passed through
the ``TabMlp`` model. Required to slice the tensors. e.g. {'education':
0, 'relationship': 1, 'workclass': 2, ...}
---
    embedding dimension. e.g. [(education, 11, 32), ...]
    Categorical embeddings dropout
    `'tanh'`, `'relu'`, `'leaky_relu'` and `'gelu'` are supported
    List with the name of the numeric (aka continuous) columns
    are: 'layernorm', 'batchnorm' or None.
    (i.e. passed each through a linear layer with or without activation)
    Size of the continuous embeddings
    Dropout for the continuous embeddings
    Boolean indicating if bias will be used for the continuous embeddings
    `'tanh'`, `'relu'`, `'leaky_relu'` and `'gelu'` are supported
    List with the number of neurons per dense layer in the mlp.
    `'tanh'`, `'relu'`, `'leaky_relu'` and `'gelu'` are supported
    distribution.
    distribution for each Bayesian linear and embedding layer
    layer
    ``posterior_rho_init`` and std equal to 0.1.
    0.1.

Attributes
----------
    This is the module that processes the categorical and continuous columns
    the continuous columns

Example
--------

```python

>>> from pytorch_widedeep.bayesian_models import BayesianTabMlp
>>> X_tab = torch.cat((torch.empty(5, 4).random_(4), torch.rand(5, 1)), axis=1)
>>> colnames = ['a', 'b', 'c', 'd', 'e']
>>> cat_embed_input = [(u,i,j) for u,i,j in zip(colnames[:4], [4]*4, [8]*4)]
>>> column_idx = {k:v for v,k in enumerate(colnames)}
>>> model = BayesianTabMlp(mlp_hidden_dims=[8,4], column_idx=column_idx, cat_embed_input=cat_embed_input,
... continuous_cols = ['e'])
>>> out = model(X_tab)
```


**Methods:**


### .__init__
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/bayesian_models/tabular/bayesian_mlp/bayesian_tab_mlp.py/#L123)
```python
.__init__(
   column_idx: Dict[str, int], cat_embed_input: Optional[List[Tuple[str, int,
   int]]] = None, cat_embed_dropout: float = 0.1,
   cat_embed_activation: Optional[str] = None,
   continuous_cols: Optional[List[str]] = None, embed_continuous: bool = False,
   cont_embed_dim: int = 32, cont_embed_dropout: float = 0.1,
   cont_embed_activation: Optional[str] = None, use_cont_bias: bool = True,
   cont_norm_layer: str = 'batchnorm', mlp_hidden_dims: List[int] = [200, 100],
   mlp_activation: str = 'leaky_relu', prior_sigma_1: float = 1,
   prior_sigma_2: float = 0.002, prior_pi: float = 0.8, posterior_mu_init: float = 0.0,
   posterior_rho_init: float = -7.0, pred_dim = 1
)
```


### .forward
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/bayesian_models/tabular/bayesian_mlp/bayesian_tab_mlp.py/#L218)
```python
.forward(
   X: Tensor
)
```

