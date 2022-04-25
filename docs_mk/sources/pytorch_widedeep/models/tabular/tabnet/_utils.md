#


### create_explain_matrix
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/models/tabular/tabnet/_utils.py/#L7)
```python
.create_explain_matrix(
   model: WideDeep
)
```

---
Returns a sparse matrix used to compute the feature importances after
training

Parameters
----------
model: WideDeep
object of type ``WideDeep``

---
Examples
--------

```python

>>> from pytorch_widedeep.models.tabular.tabnet._utils import create_explain_matrix
>>> embed_input = [("a", 4, 2), ("b", 4, 2), ("c", 4, 2)]
>>> cont_cols = ["d", "e"]
>>> column_idx = {k: v for v, k in enumerate(["a", "b", "c", "d", "e"])}
>>> deeptabular = TabNet(column_idx=column_idx, cat_embed_input=embed_input, continuous_cols=cont_cols)
>>> model = WideDeep(deeptabular=deeptabular)
>>> reduce_mtx = create_explain_matrix(model)
>>> reduce_mtx.todense()
matrix([[1., 0., 0., 0., 0.],
        [1., 0., 0., 0., 0.],
        [0., 1., 0., 0., 0.],
        [0., 1., 0., 0., 0.],
        [0., 0., 1., 0., 0.],
        [0., 0., 1., 0., 0.],
        [0., 0., 0., 1., 0.],
        [0., 0., 0., 0., 1.]])
```

----


### extract_cat_setup
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/models/tabular/tabnet/_utils.py/#L88)
```python
.extract_cat_setup(
   backbone: Module
)
```


----


### extract_cont_setup
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/models/tabular/tabnet/_utils.py/#L96)
```python
.extract_cont_setup(
   backbone: Module
)
```

