#


## SameSizeCatAndContEmbeddings
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/models/tabular/embeddings_layers.py/#L349)
```python 
SameSizeCatAndContEmbeddings(
   embed_dim: int, column_idx: Dict[str, int],
   cat_embed_input: Optional[List[Tuple[str, int]]], cat_embed_dropout: float,
   use_cat_bias: bool, full_embed_dropout: bool, shared_embed: bool,
   add_shared_embed: bool, frac_shared_embed: float,
   continuous_cols: Optional[List[str]], cont_norm_layer: str,
   embed_continuous: bool, cont_embed_dropout: float, use_cont_bias: bool
)
```




**Methods:**


### .__init__
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/models/tabular/embeddings_layers.py/#L350)
```python
.__init__(
   embed_dim: int, column_idx: Dict[str, int],
   cat_embed_input: Optional[List[Tuple[str, int]]], cat_embed_dropout: float,
   use_cat_bias: bool, full_embed_dropout: bool, shared_embed: bool,
   add_shared_embed: bool, frac_shared_embed: float,
   continuous_cols: Optional[List[str]], cont_norm_layer: str,
   embed_continuous: bool, cont_embed_dropout: float, use_cont_bias: bool
)
```


### .forward
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/models/tabular/embeddings_layers.py/#L404)
```python
.forward(
   X: Tensor
)
```

