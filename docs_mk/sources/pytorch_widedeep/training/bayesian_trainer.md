#


## BayesianTrainer
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/training/bayesian_trainer.py/#L34)
```python 
BayesianTrainer(
   model: BaseBayesianModel, objective: str,
   custom_loss_function: Optional[Module] = None, optimizer: Optimizer = None,
   lr_scheduler: LRScheduler = None, callbacks: Optional[List[Callback]] = None,
   metrics: Optional[Union[List[Metric], List[TorchMetric]]] = None, verbose: int = 1,
   seed: int = 1, **kwargs
)
```


---
Class to set the of attributes that will be used during the
training process.

Parameters
----------
model: ``BaseBayesianModel``
An object of class ``BaseBayesianModel``
---
    Possible values are: 'binary', 'multiclass', 'regression'

    folder in the repo.
    default to ``AdamW``.
    :obj:`torch.optim.lr_scheduler.StepLR(opt, step_size=5)`)
    folder in the repo
      <https://torchmetrics.readthedocs.io/en/latest/>`_.
    Setting it to 0 will print nothing during training.
    Random seed to be used internally for train_test_split

Attributes
----------
    <https://pytorch.org/docs/stable/optim.html>`_.


**Methods:**


### .__init__
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/training/bayesian_trainer.py/#L105)
```python
.__init__(
   model: BaseBayesianModel, objective: str,
   custom_loss_function: Optional[Module] = None, optimizer: Optimizer = None,
   lr_scheduler: LRScheduler = None, callbacks: Optional[List[Callback]] = None,
   metrics: Optional[Union[List[Metric], List[TorchMetric]]] = None, verbose: int = 1,
   seed: int = 1, **kwargs
)
```


### .predict
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/training/bayesian_trainer.py/#L257)
```python
.predict(
   X_tab: np.ndarray, n_samples: int = 5, return_samples: bool = False,
   batch_size: int = 256
)
```

---
Returns the predictions

Parameters
----------
X_tab: np.ndarray,
tabular dataset
---
    produce an overal prediction
    Boolean indicating whether the n samples will be averaged or directly returned
    batch size

### .save
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/training/bayesian_trainer.py/#L330)
```python
.save(
   path: str, save_state_dict: bool = False, model_filename: str = 'wd_model.pt'
)
```

---
Saves the model, training and evaluation history, and the
``feature_importance`` attribute (if the ``deeptabular`` component is a
Tabnet model) to disk

The ``Trainer`` class is built so that it 'just' trains a model. With
that in mind, all the torch related parameters (such as optimizers or
learning rate schedulers) have to be defined externally and then
passed to the ``Trainer``. As a result, the ``Trainer`` does not
generate any attribute or additional data products that need to be
saved other than the ``model`` object itself, which can be saved as
any other torch model (e.g. ``torch.save(model, path)``).

Parameters
----------
path: str
path to the directory where the model and the feature importance
attribute will be saved.
---
    model's state dictionary
    filename where the model weights will be store

### ._set_device_and_num_workers
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/training/bayesian_trainer.py/#L593)
```python
._set_device_and_num_workers(
   **kwargs
)
```


### ._set_loss_fn
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/training/bayesian_trainer.py/#L530)
```python
._set_loss_fn(
   objective, custom_loss_function, **kwargs
)
```


### .predict_proba
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/training/bayesian_trainer.py/#L290)
```python
.predict_proba(
   X_tab: np.ndarray, n_samples: int = 5, return_samples: bool = False,
   batch_size: int = 256
)
```

---
Returns the predicted probabilities

Parameters
----------
X_tab: np.ndarray,
tabular dataset
---
    produce an overal prediction
    Boolean indicating whether the n samples will be averaged or directly returned
    batch size

### ._train_step
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/training/bayesian_trainer.py/#L410)
```python
._train_step(
   X_tab: Tensor, target: Tensor, n_samples: int, n_batches: int, batch_idx: int
)
```


### ._predict
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/training/bayesian_trainer.py/#L481)
```python
._predict(
   X_tab: np.ndarray = None, n_samples: int = 5, return_samples: bool = False,
   batch_size: int = 256
)
```


### ._eval_step
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/training/bayesian_trainer.py/#L439)
```python
._eval_step(
   X_tab: Tensor, target: Tensor, n_samples: int, n_batches: int, batch_idx: int
)
```


### ._restore_best_weights
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/training/bayesian_trainer.py/#L381)
```python
._restore_best_weights()
```


### ._get_score
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/training/bayesian_trainer.py/#L469)
```python
._get_score(
   y_pred, y
)
```


### ._set_reduce_on_plateau_criterion
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/training/bayesian_trainer.py/#L546)
```python
._set_reduce_on_plateau_criterion(
   lr_scheduler, reducelronplateau_criterion
)
```


### ._set_callbacks_and_metrics
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/training/bayesian_trainer.py/#L576)
```python
._set_callbacks_and_metrics(
   callbacks, metrics
)
```


### ._set_lr_scheduler_running_params
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/training/bayesian_trainer.py/#L566)
```python
._set_lr_scheduler_running_params(
   lr_scheduler, **kwargs
)
```


### .fit
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/training/bayesian_trainer.py/#L145)
```python
.fit(
   X_tab: np.ndarray, target: np.ndarray, X_tab_val: Optional[np.ndarray] = None,
   target_val: Optional[np.ndarray] = None, val_split: Optional[float] = None,
   n_epochs: int = 1, val_freq: int = 1, batch_size: int = 32, n_train_samples: int = 2,
   n_val_samples: int = 2
)
```

---
Fit method.

Parameters
----------
X_tab: np.ndarray,
tabular dataset
---
    target values
    validation data
    validation target values
    split fraction via 'val_split'
    number of epochs
    epochs validation frequency
    batch size
    number of samples to average over during the training process.
    number of samples to average over during the validation process.
