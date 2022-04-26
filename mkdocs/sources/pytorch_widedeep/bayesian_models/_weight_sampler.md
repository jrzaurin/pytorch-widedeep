#


## GaussianPosterior
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/bayesian_models/_weight_sampler.py/#L31)
```python 
GaussianPosterior(
   param_mu: Tensor, param_rho: Tensor
)
```


---
Defines the Gaussian variational posterior as proposed in Weight
Uncertainty in Neural Networks


**Methods:**


### .sample
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/bayesian_models/_weight_sampler.py/#L46)
```python
.sample()
```


### .__init__
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/bayesian_models/_weight_sampler.py/#L36)
```python
.__init__(
   param_mu: Tensor, param_rho: Tensor
)
```


### .sigma
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/bayesian_models/_weight_sampler.py/#L43)
```python
.sigma()
```


### .log_posterior
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/bayesian_models/_weight_sampler.py/#L50)
```python
.log_posterior(
   input: Tensor
)
```

