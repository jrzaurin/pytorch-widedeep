#


### tabular_train_val_split
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/training/_trainer_utils.py/#L32)
```python
.tabular_train_val_split(
   seed: int, method: str, X: np.ndarray, y: np.ndarray,
   X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
   val_split: Optional[float] = None
)
```

---
Function to create the train/val split for the BayesianTrainer where only
tabular data is used

Parameters
----------
seed: int
random seed to be used during train/val split
---
    'regression',  'binary' or 'multiclass'
    tabular dataset (categorical and continuous features)
y: np.ndarray
    (e.g: 'wide') and the values the corresponding arrays
y_val: np.ndarray, Optional, default = None

Returns
-------
train_set: ``TensorDataset``
eval_set: ``TensorDataset``

----


### wd_train_val_split
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/training/_trainer_utils.py/#L106)
```python
.wd_train_val_split(
   seed: int, method: str, X_wide: Optional[np.ndarray] = None,
   X_tab: Optional[np.ndarray] = None, X_text: Optional[np.ndarray] = None,
   X_img: Optional[np.ndarray] = None, X_train: Optional[Dict[str,
   np.ndarray]] = None, X_val: Optional[Dict[str, np.ndarray]] = None,
   val_split: Optional[float] = None, target: Optional[np.ndarray] = None,
   transforms: Optional[List[Transforms]] = None, **lds_args
)
```

---
Function to create the train/val split for a wide and deep model

If a validation set (X_val) is passed to the fit method, or val_split is
specified, the train/val split will happen internally. A number of options
are allowed in terms of data inputs. For parameter information, please,
see the ``Trainer``'s' ``.fit()`` method documentation

Parameters
----------
seed: int
random seed to be used during train/val split
---
    'regression',  'binary' or 'multiclass'
    wide dataset
    tabular dataset (categorical and continuous features)
    image dataset
    text dataset
    (e.g: 'wide') and the values the corresponding arrays
    Alternatively, the validation split can be specified via a float
target: np.ndarray, Optional, default = None
    List of Transforms to be applied to the image dataset

Returns
-------
train_set: ``WideDeepDataset``
eval_set: ``WideDeepDataset``

----


### _build_train_dict
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/training/_trainer_utils.py/#L205)
```python
._build_train_dict(
   X_wide, X_tab, X_text, X_img, target
)
```


----


### print_loss_and_metric
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/training/_trainer_utils.py/#L218)
```python
.print_loss_and_metric(
   pb: tqdm, loss: float, score: Dict
)
```

---
Function to improve readability and avoid code repetition in the
training/validation loop within the Trainer's fit method

Parameters
----------
pb: tqdm
tqdm object defined as trange(...)
---
    Loss value
    corresponding values

----


### save_epoch_logs
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/training/_trainer_utils.py/#L244)
```python
.save_epoch_logs(
   epoch_logs: Dict, loss: float, score: Dict, stage: str
)
```

---
Function to improve readability and avoid code repetition in the
training/validation loop within the Trainer's fit method

Parameters
----------
epoch_logs: Dict
Dict containing the epoch logs
---
    loss value
    corresponding values
    one of 'train' or 'val'

----


### bayesian_alias_to_loss
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/training/_trainer_utils.py/#L269)
```python
.bayesian_alias_to_loss(
   loss_fn: str, **kwargs
)
```

---
Function that returns the corresponding loss function given an alias.
To be used with the ``BayesianTrainer``

Parameters
----------
loss_fn: str
Loss name

---
Returns
-------
Object
    loss function

Examples
--------

```python

>>> loss_fn = bayesian_alias_to_loss(loss_fn="binary", weight=None)
```

----


### alias_to_loss
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/training/_trainer_utils.py/#L297)
```python
.alias_to_loss(
   loss_fn: str, **kwargs
)
```

---
Function that returns the corresponding loss function given an alias

Parameters
----------
loss_fn: str
Loss name or alias

---
Returns
-------
Object
    loss function

Examples
--------

```python

>>> loss_fn = alias_to_loss(loss_fn="binary_logloss", weight=None)
```
