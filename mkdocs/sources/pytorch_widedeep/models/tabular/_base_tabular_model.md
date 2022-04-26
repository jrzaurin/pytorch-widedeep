#


## BaseTabularModelWithAttention
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/models/tabular/_base_tabular_model.py/#L81)
```python 
BaseTabularModelWithAttention(
   column_idx: Dict[str, int], cat_embed_input: Optional[List[Tuple[str, int]]],
   cat_embed_dropout: float, use_cat_bias: bool,
   cat_embed_activation: Optional[str], full_embed_dropout: bool,
   shared_embed: bool, add_shared_embed: bool, frac_shared_embed: float,
   continuous_cols: Optional[List[str]], cont_norm_layer: str,
   embed_continuous: bool, cont_embed_dropout: float, use_cont_bias: bool,
   cont_embed_activation: Optional[str], input_dim: int
)
```




**Methods:**


### .__init__
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/models/tabular/_base_tabular_model.py/#L82)
```python
.__init__(
   column_idx: Dict[str, int], cat_embed_input: Optional[List[Tuple[str, int]]],
   cat_embed_dropout: float, use_cat_bias: bool,
   cat_embed_activation: Optional[str], full_embed_dropout: bool,
   shared_embed: bool, add_shared_embed: bool, frac_shared_embed: float,
   continuous_cols: Optional[List[str]], cont_norm_layer: str,
   embed_continuous: bool, cont_embed_dropout: float, use_cont_bias: bool,
   cont_embed_activation: Optional[str], input_dim: int
)
```


### ._get_embeddings
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/models/tabular/_base_tabular_model.py/#L149)
```python
._get_embeddings(
   X: Tensor
)
```


### .attention_weights
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/models/tabular/_base_tabular_model.py/#L164)
```python
.attention_weights()
```

