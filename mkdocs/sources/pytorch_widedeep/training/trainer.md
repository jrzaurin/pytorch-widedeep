#


## Trainer
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/training/trainer.py/#L48)
```python 
Trainer(
   model: WideDeep, objective: str, custom_loss_function: Optional[Module] = None,
   optimizers: Optional[Union[Optimizer, Dict[str, Optimizer]]] = None,
   lr_schedulers: Optional[Union[LRScheduler, Dict[str, LRScheduler]]] = None,
   initializers: Optional[Union[Initializer, Dict[str, Initializer]]] = None,
   transforms: Optional[List[Transforms]] = None,
   callbacks: Optional[List[Callback]] = None, metrics: Optional[Union[List[Metric],
   List[TorchMetric]]] = None, verbose: int = 1, seed: int = 1, **kwargs
)
```


---
Class to set the of attributes that will be used during the
training process.

Parameters
----------
model: ``WideDeep``
An object of class ``WideDeep``
---
    - ``tweedie``
    consistent with the loss function
    model components
      values are the corresponding learning rate schedulers.
      and the values are the corresponding initializers.
    <https://pytorch.org/docs/stable/torchvision/transforms.html>`_.
    in the repo
      <https://torchmetrics.readthedocs.io/en/latest/>`_.
    Verbosity level. If set to 0 nothing will be printed during training
    Random seed to be used internally for train/test split

Attributes
----------
    <https://pytorch.org/docs/stable/optim.html>`_.
    if the ``deeptabular`` component is a Tabnet model

Examples
--------

```python

>>> from torchvision.transforms import ToTensor
>>>
>>> # wide deep imports
>>> from pytorch_widedeep.callbacks import EarlyStopping, LRHistory
>>> from pytorch_widedeep.initializers import KaimingNormal, KaimingUniform, Normal, Uniform
>>> from pytorch_widedeep.models import TabResnet, Vision, BasicRNN, Wide, WideDeep
>>> from pytorch_widedeep import Trainer
>>>
>>> embed_input = [(u, i, j) for u, i, j in zip(["a", "b", "c"][:4], [4] * 3, [8] * 3)]
>>> column_idx = {k: v for v, k in enumerate(["a", "b", "c"])}
>>> wide = Wide(10, 1)
>>>
>>> # build the model
>>> deeptabular = TabResnet(blocks_dims=[8, 4], column_idx=column_idx, cat_embed_input=embed_input)
>>> deeptext = BasicRNN(vocab_size=10, embed_dim=4, padding_idx=0)
>>> deepimage = Vision()
>>> model = WideDeep(wide=wide, deeptabular=deeptabular, deeptext=deeptext, deepimage=deepimage)
>>>
>>> # set optimizers and schedulers
>>> wide_opt = torch.optim.Adam(model.wide.parameters())
>>> deep_opt = torch.optim.AdamW(model.deeptabular.parameters())
>>> text_opt = torch.optim.Adam(model.deeptext.parameters())
>>> img_opt = torch.optim.AdamW(model.deepimage.parameters())
>>>
>>> wide_sch = torch.optim.lr_scheduler.StepLR(wide_opt, step_size=5)
>>> deep_sch = torch.optim.lr_scheduler.StepLR(deep_opt, step_size=3)
>>> text_sch = torch.optim.lr_scheduler.StepLR(text_opt, step_size=5)
>>> img_sch = torch.optim.lr_scheduler.StepLR(img_opt, step_size=3)
>>>
>>> optimizers = {"wide": wide_opt, "deeptabular": deep_opt, "deeptext": text_opt, "deepimage": img_opt}
>>> schedulers = {"wide": wide_sch, "deeptabular": deep_sch, "deeptext": text_sch, "deepimage": img_sch}
>>>
>>> # set initializers and callbacks
>>> initializers = {"wide": Uniform, "deeptabular": Normal, "deeptext": KaimingNormal, "deepimage": KaimingUniform}
>>> transforms = [ToTensor]
>>> callbacks = [LRHistory(n_epochs=4), EarlyStopping]
>>>
>>> # set the trainer
>>> trainer = Trainer(model, objective="regression", initializers=initializers, optimizers=optimizers,
... lr_schedulers=schedulers, callbacks=callbacks, transforms=transforms)
```


**Methods:**


### .__init__
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/training/trainer.py/#L209)
```python
.__init__(
   model: WideDeep, objective: str, custom_loss_function: Optional[Module] = None,
   optimizers: Optional[Union[Optimizer, Dict[str, Optimizer]]] = None,
   lr_schedulers: Optional[Union[LRScheduler, Dict[str, LRScheduler]]] = None,
   initializers: Optional[Union[Initializer, Dict[str, Initializer]]] = None,
   transforms: Optional[List[Transforms]] = None,
   callbacks: Optional[List[Callback]] = None, metrics: Optional[Union[List[Metric],
   List[TorchMetric]]] = None, verbose: int = 1, seed: int = 1, **kwargs
)
```


### .explain
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/training/trainer.py/#L756)
```python
.explain(
   X_tab: np.ndarray, save_step_masks: bool = False
)
```

---
if the ``deeptabular`` component is a :obj:`Tabnet` model, returns the
aggregated feature importance for each instance (or observation) in
the ``X_tab`` array. If ``save_step_masks`` is set to ``True``, the
masks per step will also be returned.

Parameters
----------
X_tab: np.ndarray
Input array corresponding **only** to the deeptabular component
---
    Boolean indicating if the masks per step will be returned

Returns
-------
    is set to ``True``

### .fit
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/training/trainer.py/#L254)
```python
.fit(
   X_wide: Optional[np.ndarray] = None, X_tab: Optional[np.ndarray] = None,
   X_text: Optional[np.ndarray] = None, X_img: Optional[np.ndarray] = None,
   X_train: Optional[Dict[str, np.ndarray]] = None, X_val: Optional[Dict[str,
   np.ndarray]] = None, val_split: Optional[float] = None,
   target: Optional[np.ndarray] = None, n_epochs: int = 1, validation_freq: int = 1,
   batch_size: int = 32, custom_dataloader: Optional[DataLoader] = None,
   finetune: bool = False, with_lds: bool = False, **kwargs
)
```

---
Fit method.

The input datasets can be passed either directly via numpy arrays
(``X_wide``, ``X_tab``, ``X_text`` or ``X_img``) or alternatively, in
dictionaries (``X_train`` or ``X_val``).

Parameters
----------
X_wide: np.ndarray, Optional. default=None
Input for the ``wide`` model component.
See :class:`pytorch_widedeep.preprocessing.WidePreprocessor`
---
    See :class:`pytorch_widedeep.preprocessing.TabPreprocessor`
    See :class:`pytorch_widedeep.preprocessing.TextPreprocessor`
    See :class:`pytorch_widedeep.preprocessing.ImagePreprocessor`
    are the corresponding matrices.
    Values are the corresponding matrices.
    train/val split fraction
    target values
    number of epochs
    epochs validation frequency
    batch size
    ``None``, a standard torch ``DataLoader`` is used.
    folder in the repo.

Examples
--------

For a series of comprehensive examples please, see the `Examples
<https://github.com/jrzaurin/pytorch-widedeep/tree/master/examples>`__
folder in the repo

For completion, here we include some `"fabricated"` examples, i.e.
these assume you have already built a model and instantiated a
``Trainer``, that is ready to fit

    trainer.fit(X_wide=X_wide, X_tab=X_tab, target=target, n_epochs=10, batch_size=256)


    trainer.fit(X_wide=X_wide, X_tab=X_tab, target=target, n_epochs=10, batch_size=256, val_split=0.2)


    trainer.fit(X_train, n_epochs=10, batch_size=256, val_split=0.2)


    trainer.fit(X_train=X_train, X_val=X_val n_epochs=10, batch_size=256)

### ._get_score
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/training/trainer.py/#L1069)
```python
._get_score(
   y_pred, y
)
```


### ._finetune
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/training/trainer.py/#L910)
```python
._finetune(
   loader: DataLoader, n_epochs: int = 5, max_lr: float = 0.01,
   routine: Literal['howard', 'felbo'] = 'howard', deeptabular_gradual: bool = False,
   deeptabular_layers: Optional[List[nn.Module]] = None,
   deeptabular_max_lr: float = 0.01, deeptext_gradual: bool = False,
   deeptext_layers: Optional[List[nn.Module]] = None, deeptext_max_lr: float = 0.01,
   deepimage_gradual: bool = False,
   deepimage_layers: Optional[List[nn.Module]] = None, deepimage_max_lr: float = 0.01
)
```

---
Simple wrap-up to individually fine-tune model components

### ._set_callbacks_and_metrics
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/training/trainer.py/#L1412)
```python
._set_callbacks_and_metrics(
   callbacks, metrics
)
```


### ._compute_feature_importance
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/training/trainer.py/#L1114)
```python
._compute_feature_importance(
   loader: DataLoader
)
```


### ._extract_kwargs
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/training/trainer.py/#L1237)
```python
._extract_kwargs(
   kwargs
)
```


### ._set_transforms
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/training/trainer.py/#L1406)
```python
._set_transforms(
   transforms
)
```


### .predict_uncertainty
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/training/trainer.py/#L552)
```python
.predict_uncertainty(
   X_wide: Optional[np.ndarray] = None, X_tab: Optional[np.ndarray] = None,
   X_text: Optional[np.ndarray] = None, X_img: Optional[np.ndarray] = None,
   X_test: Optional[Dict[str, np.ndarray]] = None, batch_size: int = 256,
   uncertainty_granularity = 1000
)
```

---
Returns the predicted ucnertainty of the model for the test dataset
using a Monte Carlo method during which dropout layers are activated
in the evaluation/prediction phase and each sample is predicted N
times (``uncertainty_granularity`` times).

This is based on `'Gal Y. & Ghahramani Z., 2016, Dropout as a Bayesian
Approximation: Representing Model Uncertainty in Deep Learning'`,
Proceedings of the 33rd International Conference on Machine Learning

Parameters
----------
X_wide: np.ndarray, Optional. default=None
Input for the ``wide`` model component.
See :class:`pytorch_widedeep.preprocessing.WidePreprocessor`
---
    See :class:`pytorch_widedeep.preprocessing.TabPreprocessor`
    See :class:`pytorch_widedeep.preprocessing.TextPreprocessor`
    See :class:`pytorch_widedeep.preprocessing.ImagePreprocessor`
    are the corresponding matrices.
    the :obj:`Trainer` is instantiated
    is set to True

Returns
-------
    - if ``method == regression``, it will return an array with `{max, min, mean, stdev}`
      values for each sample.
    - if ``method == binary`` it will return an array with
      `{mean_cls_0_prob, mean_cls_1_prob, predicted_cls}` for each sample.
    - if ``method == multiclass`` it will return an array with
      `{mean_cls_0_prob, mean_cls_1_prob, mean_cls_2_prob, ... , predicted_cls}` values for each sample.

### .save
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/training/trainer.py/#L816)
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
that in mind, all the torch related parameters (such as optimizers,
learning rate schedulers, initializers, etc) have to be defined
externally and then passed to the ``Trainer``. As a result, the
``Trainer`` does not generate any attribute or additional data
products that need to be saved other than the ``model`` object itself,
which can be saved as any other torch model (e.g. ``torch.save(model,
path)``).

The exception is Tabnet. If the ``deeptabular`` component is a Tabnet
model, an attribute (a dict) called ``feature_importance`` will be
created at the end of the training process. Therefore, a ``save``
method was created that will save the feature importance dictionary
to a json file and, since we are here, the model weights, training
history and learning rate history.

Parameters
----------
path: str
path to the directory where the model and the feature importance
attribute will be saved.
---
    model's state dictionary
    filename where the model weights will be store

### ._predict_ziln
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/training/trainer.py/#L1219)
```python
._predict_ziln(
   preds: Tensor
)
```

---
Calculates predicted mean of zero inflated lognormal logits.

Adjusted implementaion of `code
<https://github.com/google/lifetime_value/blob/master/lifetime_value/zero_inflated_lognormal.py>`


**Arguments**

* **preds**  : [batch_size, 3] tensor of logits.


**Returns**

* **ziln_preds**  : [batch_size, 1] tensor of predicted mean.


### ._predict
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/training/trainer.py/#L1136)
```python
._predict(
   X_wide: Optional[np.ndarray] = None, X_tab: Optional[np.ndarray] = None,
   X_text: Optional[np.ndarray] = None, X_img: Optional[np.ndarray] = None,
   X_test: Optional[Dict[str, np.ndarray]] = None, batch_size: int = 256,
   uncertainty_granularity = 1000, uncertainty: bool = False, quantiles: bool = False
)
```

---
Private method to avoid code repetition in predict and
predict_proba. For parameter information, please, see the .predict()
method documentation

### ._set_device_and_num_workers
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/training/trainer.py/#L1468)
```python
._set_device_and_num_workers(
   **kwargs
)
```


### .predict
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/training/trainer.py/#L501)
```python
.predict(
   X_wide: Optional[np.ndarray] = None, X_tab: Optional[np.ndarray] = None,
   X_text: Optional[np.ndarray] = None, X_img: Optional[np.ndarray] = None,
   X_test: Optional[Dict[str, np.ndarray]] = None, batch_size: int = 256
)
```

---
Returns the predictions

The input datasets can be passed either directly via numpy arrays
(``X_wide``, ``X_tab``, ``X_text`` or ``X_img``) or alternatively, in
a dictionary (``X_test``)


Parameters
----------
X_wide: np.ndarray, Optional. default=None
Input for the ``wide`` model component.
See :class:`pytorch_widedeep.preprocessing.WidePreprocessor`
---
    See :class:`pytorch_widedeep.preprocessing.TabPreprocessor`
    See :class:`pytorch_widedeep.preprocessing.TextPreprocessor`
    See :class:`pytorch_widedeep.preprocessing.ImagePreprocessor`
    are the corresponding matrices.
    the :obj:`Trainer` is instantiated

### ._check_inputs
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/training/trainer.py/#L1429)
```python
._check_inputs(
   model, objective, optimizers, lr_schedulers, custom_loss_function
)
```


### .get_embeddings
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/training/trainer.py/#L698)
```python
.get_embeddings(
   col_name: str, cat_encoding_dict: Dict[str, Dict[str, int]]
)
```

---
Returns the learned embeddings for the categorical features passed through
``deeptabular``.

.. note:: This function will be deprecated in the next relase. Please consider
using ``Tab2Vec`` instead.

---
This method is designed to take an encoding dictionary in the same
format as that of the :obj:`LabelEncoder` Attribute in the class
:obj:`TabPreprocessor`. See
:class:`pytorch_widedeep.preprocessing.TabPreprocessor` and
:class:`pytorch_widedeep.utils.dense_utils.LabelEncder`.

Parameters
----------
    Column name of the feature we want to get the embeddings for
    e.g.: {'column': {'cat_0': 1, 'cat_1': 2, ...}}

Examples
--------

For a series of comprehensive examples please, see the `Examples
<https://github.com/jrzaurin/pytorch-widedeep/tree/master/examples>`__
folder in the repo

For completion, here we include a `"fabricated"` example, i.e.
assuming we have already trained the model, that we have the
categorical encodings in a dictionary name ``encoding_dict``, and that
there is a column called `'education'`:

    trainer.get_embeddings(col_name="education", cat_encoding_dict=encoding_dict)

### ._train_step
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/training/trainer.py/#L983)
```python
._train_step(
   data: Dict[str, Tensor], target: Tensor, batch_idx: int, epoch: int,
   lds_weightt: Tensor
)
```


### ._eval_step
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/training/trainer.py/#L1044)
```python
._eval_step(
   data: Dict[str, Tensor], target: Tensor, batch_idx: int
)
```


### ._restore_best_weights
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/training/trainer.py/#L879)
```python
._restore_best_weights()
```


### ._set_loss_fn
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/training/trainer.py/#L1305)
```python
._set_loss_fn(
   objective, custom_loss_function, **kwargs
)
```


### ._set_reduce_on_plateau_criterion
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/training/trainer.py/#L1350)
```python
._set_reduce_on_plateau_criterion(
   lr_schedulers, reducelronplateau_criterion
)
```


### ._fds_step
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/training/trainer.py/#L1083)
```python
._fds_step(
   data: Dict[str, Tensor], target: Tensor, epoch: int
)
```


### ._set_optimizer
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/training/trainer.py/#L1327)
```python
._set_optimizer(
   optimizers
)
```


### ._update_fds_stats
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/training/trainer.py/#L1095)
```python
._update_fds_stats(
   train_loader: DataLoader, epoch: int
)
```


### ._initialize
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/training/trainer.py/#L1291)
```python
._initialize(
   initializers
)
```


### ._set_lr_scheduler
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/training/trainer.py/#L1375)
```python
._set_lr_scheduler(
   lr_schedulers, **kwargs
)
```


### .predict_proba
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/training/trainer.py/#L648)
```python
.predict_proba(
   X_wide: Optional[np.ndarray] = None, X_tab: Optional[np.ndarray] = None,
   X_text: Optional[np.ndarray] = None, X_img: Optional[np.ndarray] = None,
   X_test: Optional[Dict[str, np.ndarray]] = None, batch_size: int = 256
)
```

---
Returns the predicted probabilities for the test dataset for  binary
and multiclass methods

The input datasets can be passed either directly via numpy arrays
(``X_wide``, ``X_tab``, ``X_text`` or ``X_img``) or alternatively, in
a dictionary (``X_test``)

Parameters
----------
X_wide: np.ndarray, Optional. default=None
Input for the ``wide`` model component.
See :class:`pytorch_widedeep.preprocessing.WidePreprocessor`
---
    See :class:`pytorch_widedeep.preprocessing.TabPreprocessor`
    See :class:`pytorch_widedeep.preprocessing.TextPreprocessor`
    See :class:`pytorch_widedeep.preprocessing.ImagePreprocessor`
    are the corresponding matrices.
    the :obj:`Trainer` is instantiated
