#


## RayTuneReporter
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/callbacks.py/#L709)
```python 
RayTuneReporter()
```


---
Callback that allows reporting history and lr_history values to RayTune
during Hyperparameter tuning

Callbacks are passed as input parameters to the ``Trainer`` class. See
:class:`pytorch_widedeep.trainer.Trainer`

For examples see the examples folder at:

    /examples/12_HyperParameter_tuning_w_RayTune.ipynb


**Methods:**


### .on_epoch_end
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/callbacks.py/#L723)
```python
.on_epoch_end(
   epoch: int, logs: Optional[Dict] = None, metric: Optional[float] = None
)
```


----


### _get_current_time
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/callbacks.py/#L19)
```python
._get_current_time()
```


----


### _is_metric
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/callbacks.py/#L23)
```python
._is_metric(
   monitor: str
)
```

