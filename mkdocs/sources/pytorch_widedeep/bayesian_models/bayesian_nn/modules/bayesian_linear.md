#


## BayesianLinear
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/bayesian_models/bayesian_nn/modules/bayesian_linear.py/#L24)
```python 
BayesianLinear(
   in_features: int, out_features: int, use_bias: bool = True,
   prior_sigma_1: float = 1.0, prior_sigma_2: float = 0.002, prior_pi: float = 0.8,
   posterior_mu_init: float = 0.0, posterior_rho_init: float = -7.0
)
```


---
Applies a linear transformation to the incoming data as proposed in Weight
Uncertainity on Neural Networks

Parameters
----------
in_features: int
size of each input sample
---
     size of each output sample
    Boolean indicating if an additive bias will be learnt
    distribution
    distribution
    prior weight distribution
    ``posterior_rho_init`` and std equal to 0.1.
    0.1.

Examples
--------

```python

>>> from pytorch_widedeep.bayesian_models import bayesian_nn as bnn
>>> linear = bnn.BayesianLinear(10, 6)
>>> input = torch.rand(6, 10)
>>> out = linear(input)
```


**Methods:**


### .__init__
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/bayesian_models/bayesian_nn/modules/bayesian_linear.py/#L79)
```python
.__init__(
   in_features: int, out_features: int, use_bias: bool = True,
   prior_sigma_1: float = 1.0, prior_sigma_2: float = 0.002, prior_pi: float = 0.8,
   posterior_mu_init: float = 0.0, posterior_rho_init: float = -7.0
)
```


### .forward
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/bayesian_models/bayesian_nn/modules/bayesian_linear.py/#L140)
```python
.forward(
   X: Tensor
)
```


### .extra_repr
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/bayesian_models/bayesian_nn/modules/bayesian_linear.py/#L164)
```python
.extra_repr()
```

