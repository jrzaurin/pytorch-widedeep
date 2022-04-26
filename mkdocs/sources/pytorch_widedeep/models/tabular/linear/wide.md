#


## Wide
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/models/tabular/linear/wide.py/#L9)
```python 
Wide(
   input_dim: int, pred_dim: int = 1
)
```


---
Defines a ``Wide`` (linear) model where the non-linearities are
captured via the so-called crossed-columns. This can be used as the
``wide`` component of a Wide & Deep model.

Parameters
-----------
input_dim: int
size of the Embedding layer. `input_dim` is the summation of all the
individual values for all the features that go through the wide
model. For example, if the wide model receives 2 features with
5 individual values each, `input_dim = 10`
---
    it requires the ``pred_dim`` parameter.

Attributes
-----------
    the linear layer that comprises the wide branch of the model

Examples
--------

```python

>>> from pytorch_widedeep.models import Wide
>>> X = torch.empty(4, 4).random_(6)
>>> wide = Wide(input_dim=X.unique().size(0), pred_dim=1)
>>> out = wide(X)
```


**Methods:**


### .__init__
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/models/tabular/linear/wide.py/#L41)
```python
.__init__(
   input_dim: int, pred_dim: int = 1
)
```


### .forward
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/models/tabular/linear/wide.py/#L63)
```python
.forward(
   X: Tensor
)
```

---
Forward pass. Simply connecting the Embedding layer with the ouput
neuron(s)

### ._reset_parameters
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/models/tabular/linear/wide.py/#L53)
```python
._reset_parameters()
```

---
initialize Embedding and bias like nn.Linear. See `original
implementation
<https://pytorch.org/docs/stable/_modules/torch/nn/modules/linear.html#Linear>`_.
