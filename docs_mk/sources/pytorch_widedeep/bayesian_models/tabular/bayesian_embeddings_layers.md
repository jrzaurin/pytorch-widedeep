#


## BayesianDiffSizeCatAndContEmbeddings
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/bayesian_models/tabular/bayesian_embeddings_layers.py/#L147)
```python 
BayesianDiffSizeCatAndContEmbeddings(
   column_idx: Dict[str, int], cat_embed_input: List[Tuple[str, int, int]],
   continuous_cols: Optional[List[str]], embed_continuous: bool,
   cont_embed_dim: int, use_cont_bias: bool, cont_norm_layer: Optional[str],
   prior_sigma_1: float, prior_sigma_2: float, prior_pi: float,
   posterior_mu_init: float, posterior_rho_init: float
)
```




**Methods:**


### .__init__
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/bayesian_models/tabular/bayesian_embeddings_layers.py/#L148)
```python
.__init__(
   column_idx: Dict[str, int], cat_embed_input: List[Tuple[str, int, int]],
   continuous_cols: Optional[List[str]], embed_continuous: bool,
   cont_embed_dim: int, use_cont_bias: bool, cont_norm_layer: Optional[str],
   prior_sigma_1: float, prior_sigma_2: float, prior_pi: float,
   posterior_mu_init: float, posterior_rho_init: float
)
```


### .forward
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/bayesian_models/tabular/bayesian_embeddings_layers.py/#L213)
```python
.forward(
   X: Tensor
)
```

