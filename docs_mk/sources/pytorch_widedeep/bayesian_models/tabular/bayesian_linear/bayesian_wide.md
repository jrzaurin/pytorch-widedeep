#


## BayesianWide
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/bayesian_models/tabular/bayesian_linear/bayesian_wide.py/#L10)
```python 
BayesianWide(
   input_dim: int, pred_dim: int = 1, prior_sigma_1: float = 1.0,
   prior_sigma_2: float = 0.002, prior_pi: float = 0.8, posterior_mu_init: float = 0.0,
   posterior_rho_init: float = -7.0
)
```


---
Defines a ``Wide`` model. This is a linear model where the
non-linearlities are captured via crossed-columns

Parameters
----------
input_dim: int
size of the Embedding layer. ``input_dim`` is the summation of all the
individual values for all the features that go through the wide
component. For example, if the wide component receives 2 features with
5 individual values each, `input_dim = 10`
---
    size of the ouput tensor containing the predictions
    distribution.
    distribution
    prior weight distribution
    ``posterior_rho_init`` and std equal to 0.1.
    0.1.

Attributes
-----------
    the linear layer that comprises the wide branch of the model

Examples
--------

```python

>>> from pytorch_widedeep.bayesian_models import BayesianWide
>>> X = torch.empty(4, 4).random_(6)
>>> wide = BayesianWide(input_dim=X.unique().size(0), pred_dim=1)
>>> out = wide(X)
```


**Methods:**


### .__init__
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/bayesian_models/tabular/bayesian_linear/bayesian_wide.py/#L79)
```python
.__init__(
   input_dim: int, pred_dim: int = 1, prior_sigma_1: float = 1.0,
   prior_sigma_2: float = 0.002, prior_pi: float = 0.8, posterior_mu_init: float = 0.0,
   posterior_rho_init: float = -7.0
)
```


### .forward
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/bayesian_models/tabular/bayesian_linear/bayesian_wide.py/#L103)
```python
.forward(
   X: Tensor
)
```

