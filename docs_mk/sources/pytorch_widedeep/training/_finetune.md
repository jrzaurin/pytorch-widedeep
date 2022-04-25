#


## FineTune
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/training/_finetune.py/#L13)
```python 
FineTune(
   loss_fn: Any, metric: Union[Metric, MultipleMetrics], method: Literal['binary',
   'regression', 'multiclass'], verbose: int
)
```


---
Fine-tune methods to be applied to the individual model components.

Note that they can also be used to "warm-up" those components before
the joined training.

There are 3 fine-tune/warm-up routines available:

1) Fine-tune all trainable layers at once

2) Gradual fine-tuning inspired by the work of Felbo et al., 2017

3) Gradual fine-tuning inspired by the work of Howard & Ruder 2018

The structure of the code in this class is designed to be instantiated
within the class WideDeep. This is not ideal, but represents a
compromise towards implementing a fine-tuning functionality for the
current overall structure of the package without having to
re-structure most of the existing code. This will change in future
releases.

Parameters
----------
loss_fn: Any
   any function with the same strucure as 'loss_fn' in the class ``Trainer``
metric: ``Metric`` or ``MultipleMetrics``
   object of class Metric (see Metric in pytorch_widedeep.metrics)
method: str
   one of 'binary', 'regression' or 'multiclass'
verbose: Boolean


**Methods:**


### .__init__
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/training/_finetune.py/#L45)
```python
.__init__(
   loss_fn: Any, metric: Union[Metric, MultipleMetrics], method: Literal['binary',
   'regression', 'multiclass'], verbose: int
)
```


### ._steps_up_down
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/training/_finetune.py/#L289)
```python
._steps_up_down(
   steps: int, n_epochs: int = 1
)
```

---
Calculate the number of steps up and down during the one cycle fine-tune for a
given number of epochs

Parameters:
----------
steps: int
steps per epoch
---
    number of fine-tune epochs

Returns
-------
    number of steps increasing/decreasing the learning rate during the cycle

### .finetune_gradual
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/training/_finetune.py/#L112)
```python
.finetune_gradual(
   model: nn.Module, model_name: str, loader: DataLoader, last_layer_max_lr: float,
   layers: List[nn.Module], routine: str
)
```

---
Fine-tune/warm-up certain layers within the model following a
gradual fine-tune routine. The approaches implemented in this method are
based on fine-tuning routines described in the the work of Felbo et
al., 2017 in their DeepEmoji paper (https://arxiv.org/abs/1708.00524)
and Howard & Sebastian Ruder 2018 ULMFit paper
(https://arxiv.org/abs/1801.06146).

A one cycle triangular learning rate is used. In both Felbo's and
Howard's routines a gradually decreasing learning rate is used as we
go deeper into the network. The 'closest' layer to the output
neuron(s) will use a maximum learning rate of 'last_layer_max_lr'. The
learning rate will then decrease by a factor of 2.5 per layer

1) The 'Felbo' routine: train the first layer in 'layers' for one
   epoch. Then train the next layer in 'layers' for one epoch freezing
   the already trained up layer(s). Repeat untill all individual layers
   are trained. Then, train one last epoch with all trained/fine-tuned
   layers trainable

2) The 'Howard' routine: fine-tune the first layer in 'layers' for one
   epoch. Then traine the next layer in the model for one epoch while
   keeping the already trained up layer(s) trainable. Repeat.

Parameters:
----------
model: ``Module``
   ``Module`` object containing one the WideDeep model components (wide,
   deeptabular, deeptext or deepimage)
model_name: str
   string indicating the model name to access the corresponding parameters.
   One of 'wide', 'deeptabular', 'deeptext' or 'deepimage'
loader: ``DataLoader``
   Pytorch DataLoader containing the data to fine-tune with.
last_layer_max_lr: float
   maximum learning rate value during the triangular cycle for the layer
   closest to the output neuron(s). Deeper layers in 'model' will be trained
   with a gradually descending learning rate. The descending factor is fixed
   and is 2.5
layers: list
   List of ``Module`` objects containing the layers that will be fine-tuned.
   This must be in *'FINE-TUNE ORDER'*.
routine: str
   one of 'howard' or 'felbo'

### .finetune_all
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/training/_finetune.py/#L57)
```python
.finetune_all(
   model: nn.Module, model_name: str, loader: DataLoader, n_epochs: int,
   max_lr: float
)
```

---
Fine-tune/warm-up all trainable layers in a model using a one cyclic
learning rate with a triangular pattern. This is refereed as Slanted
Triangular learing rate in Jeremy Howard & Sebastian Ruder 2018
(https://arxiv.org/abs/1801.06146). The cycle is described as follows:

1) The learning rate will gradually increase for 10% of the training steps
from max_lr/10 to max_lr.

---
2) It will then gradually decrease to max_lr/10 for the remaining 90% of the
    steps.

The optimizer used in the process is AdamW

Parameters:
----------
    deeptabular, deeptext or deepimage)
    One of 'wide', 'deeptabular', 'deeptext' or 'deepimage'
    Pytorch DataLoader containing the data used to fine-tune
    number of epochs used to fine-tune the model
    maximum learning rate value during the triangular cycle.

### ._finetune
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/training/_finetune.py/#L237)
```python
._finetune(
   model: nn.Module, model_name: str, loader: DataLoader, optimizer: Optimizer,
   scheduler: LRScheduler, n_epochs: int = 1
)
```

---
Standard Pytorch training loop
