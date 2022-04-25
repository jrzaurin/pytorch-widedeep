#


## BaseBayesianModel
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/bayesian_models/_base_bayesian_model.py/#L16)
```python 
BaseBayesianModel()
```


---
Base model containing the two methods common to all Bayesian models


**Methods:**


### ._kl_divergence
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/bayesian_models/_base_bayesian_model.py/#L22)
```python
._kl_divergence()
```


### .sample_elbo
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/bayesian_models/_base_bayesian_model.py/#L29)
```python
.sample_elbo(
   input: Tensor, target: Tensor, loss_fn: nn.Module, n_samples: int, n_batches: int
)
```


### .init
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/bayesian_models/_base_bayesian_model.py/#L19)
```python
.init()
```

