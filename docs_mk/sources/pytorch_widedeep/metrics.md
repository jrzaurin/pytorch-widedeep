#


## R2Score
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/metrics.py/#L347)
```python 

```


---
Calculates the R-Squared, the
`coefficient of determination <https://en.wikipedia.org/wiki/Coefficient_of_determination>`_:

:math:`R^2 = 1 - \frac{\sum_{j=1}^n(y_j - \hat{y_j})^2}{\sum_{j=1}^n(y_j - \bar{y})^2}`,

where :math:`\hat{y_j}` is the ground truth, :math:`y_j` is the predicted value and
:math:`\bar{y}` is the mean of the ground truth.

Examples
--------

```python

>>>
>>> from pytorch_widedeep.metrics import R2Score
>>>
>>> r2 = R2Score()
>>> y_true = torch.tensor([3, -0.5, 2, 7]).view(-1, 1)
>>> y_pred = torch.tensor([2.5, 0.0, 2, 8]).view(-1, 1)
>>> r2(y_pred, y_true)
array(0.94860814)
```


**Methods:**


### .__init__
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/metrics.py/#L370)
```python
.__init__()
```


### .reset
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/metrics.py/#L378)
```python
.reset()
```

---
resets counters to 0

### .__call__
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/metrics.py/#L387)
```python
.__call__(
   y_pred: Tensor, y_true: Tensor
)
```

