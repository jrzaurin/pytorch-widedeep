#


## HuberLoss
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/losses.py/#L769)
```python 
HuberLoss(
   beta: float = 0.2
)
```


---
Hubbler Loss

Based on `Delving into Deep Imbalanced Regression
<https://arxiv.org/abs/2102.09554>`_ and their `implementation
<https://github.com/YyzHarry/imbalanced-regression>`_


**Methods:**


### .__init__
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/losses.py/#L777)
```python
.__init__(
   beta: float = 0.2
)
```


### .forward
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/losses.py/#L781)
```python
.forward(
   input: Tensor, target: Tensor, lds_weight: Optional[Tensor] = None
)
```

---
Parameters
----------
input: Tensor
Input tensor with predictions (not probabilities)
---
    Target tensor with the actual classes
    multiply the loss value.
    publication for details.

Examples
--------

```python

>>>
>>> from pytorch_widedeep.losses import HuberLoss
>>>
>>> target = torch.tensor([1, 1.2, 0, 2]).view(-1, 1)
>>> input = torch.tensor([0.6, 0.7, 0.3, 0.8]).view(-1, 1)
>>> HuberLoss()(input, target)
tensor(0.5000)
```
