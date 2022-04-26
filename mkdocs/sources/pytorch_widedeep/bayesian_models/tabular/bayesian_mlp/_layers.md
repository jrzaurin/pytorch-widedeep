#


## BayesianMLP
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/bayesian_models/tabular/bayesian_mlp/_layers.py/#L8)
```python 
BayesianMLP(
   d_hidden: List[int], activation: str, use_bias: bool = True,
   prior_sigma_1: float = 1.0, prior_sigma_2: float = 0.002, prior_pi: float = 0.8,
   posterior_mu_init: float = 0.0, posterior_rho_init: float = -7.0
)
```




**Methods:**


### .__init__
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/bayesian_models/tabular/bayesian_mlp/_layers.py/#L9)
```python
.__init__(
   d_hidden: List[int], activation: str, use_bias: bool = True,
   prior_sigma_1: float = 1.0, prior_sigma_2: float = 0.002, prior_pi: float = 0.8,
   posterior_mu_init: float = 0.0, posterior_rho_init: float = -7.0
)
```


### .forward
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/bayesian_models/tabular/bayesian_mlp/_layers.py/#L49)
```python
.forward(
   X: Tensor
)
```

