#


## BayesianEmbedding
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/bayesian_models/bayesian_nn/modules/bayesian_embedding.py/#L20)
```python 
BayesianEmbedding(
   n_embed: int, embed_dim: int, padding_idx: Optional[int] = None,
   max_norm: Optional[float] = None, norm_type: Optional[float] = 2.0,
   scale_grad_by_freq: Optional[bool] = False, sparse: Optional[bool] = False,
   prior_sigma_1: float = 1.0, prior_sigma_2: float = 0.002, prior_pi: float = 0.8,
   posterior_mu_init: float = 0.0, posterior_rho_init: float = -7.0
)
```


---
A simple lookup table that looks up embeddings in a fixed dictionary and
size.

Parameters
----------
n_embed: int
number of embeddings. Typically referred as size of the vocabulary
---
    Dimension of the embeddings
    used as the padding vector
    renormalized to have norm max_norm
    The p of the p-norm to compute for the ``max_norm`` option.
    words in the mini-batch.
    Notes for more details regarding sparse gradients.
    distribution
    distribution
    prior weight distribution
    ``posterior_rho_init`` and std equal to 0.1.
    0.1.

Examples
--------

```python

>>> from pytorch_widedeep.bayesian_models import bayesian_nn as bnn
>>> embedding = bnn.BayesianEmbedding(10, 3)
>>> input = torch.LongTensor([[1,2,4,5],[4,3,2,9]])
>>> out = embedding(input)
```


**Methods:**


### .__init__
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/bayesian_models/bayesian_nn/modules/bayesian_embedding.py/#L91)
```python
.__init__(
   n_embed: int, embed_dim: int, padding_idx: Optional[int] = None,
   max_norm: Optional[float] = None, norm_type: Optional[float] = 2.0,
   scale_grad_by_freq: Optional[bool] = False, sparse: Optional[bool] = False,
   prior_sigma_1: float = 1.0, prior_sigma_2: float = 0.002, prior_pi: float = 0.8,
   posterior_mu_init: float = 0.0, posterior_rho_init: float = -7.0
)
```


### .forward
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/bayesian_models/bayesian_nn/modules/bayesian_embedding.py/#L141)
```python
.forward(
   X: Tensor
)
```


### .extra_repr
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/bayesian_models/bayesian_nn/modules/bayesian_embedding.py/#L169)
```python
.extra_repr()
```

