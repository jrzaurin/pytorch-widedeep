"""
Code here is mostly based on the code from the torchsample and Keras packages

CREDIT TO THE TORCHSAMPLE AND KERAS TEAMS
"""

import os
import copy
import datetime
import warnings

import numpy as np
import torch
from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau

from pytorch_widedeep.metrics import MultipleMetrics
from pytorch_widedeep.wdtypes import Any, Dict, List, Optional, Optimizer


def _get_current_time():
    return datetime.datetime.now().strftime("%B %d, %Y - %I:%M%p")


def _is_metric(monitor: str):
    # We assume no one will use f3 or more
    if any([s in monitor for s in ["acc", "prec", "rec", "fscore", "f1", "f2"]]):
        return True
    else:
        return False


class CallbackContainer(object):
    """
    Container holding a list of callbacks.
    """

    def __init__(self, callbacks: Optional[List] = None, queue_length: int = 10):
        instantiated_callbacks = []
        if callbacks is not None:
            for callback in callbacks:
                if isinstance(callback, type):
                    instantiated_callbacks.append(callback())
                else:
                    instantiated_callbacks.append(callback)
        self.callbacks = [c for c in instantiated_callbacks]
        self.queue_length = queue_length

    def set_params(self, params):
        for callback in self.callbacks:
            callback.set_params(params)

    def set_model(self, model: Any):
        self.model = model
        for callback in self.callbacks:
            callback.set_model(model)

    def set_trainer(self, trainer: Any):
        self.trainer = trainer
        for callback in self.callbacks:
            callback.set_trainer(trainer)

    def on_epoch_begin(self, epoch: int, logs: Optional[Dict] = None):
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_epoch_begin(epoch, logs)

    def on_epoch_end(
        self, epoch: int, logs: Optional[Dict] = None, metric: Optional[float] = None
    ):
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_epoch_end(epoch, logs, metric)

    def on_batch_begin(self, batch: int, logs: Optional[Dict] = None):
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_batch_begin(batch, logs)

    def on_batch_end(self, batch: int, logs: Optional[Dict] = None):
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_batch_end(batch, logs)

    def on_train_begin(self, logs: Optional[Dict] = None):
        logs = logs or {}
        logs["start_time"] = _get_current_time()
        for callback in self.callbacks:
            callback.on_train_begin(logs)

    def on_train_end(self, logs: Optional[Dict] = None):
        logs = logs or {}
        # logs['final_loss'] = self.model.history.epoch_losses[-1],
        # logs['best_loss'] = min(self.model.history.epoch_losses),
        # logs['stop_time'] = _get_current_time()
        for callback in self.callbacks:
            callback.on_train_end(logs)

    def on_eval_begin(self, logs: Optional[Dict] = None):
        # at the moment only used to reset metrics before eval
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_eval_begin(logs)


class Callback(object):
    """
    Base class used to build new callbacks.
    """

    def __init__(self):
        pass

    def set_params(self, params):
        self.params = params

    def set_model(self, model: Any):
        self.model = model

    def set_trainer(self, trainer: Any):
        self.trainer = trainer

    def on_epoch_begin(self, epoch: int, logs: Optional[Dict] = None):
        pass

    def on_epoch_end(
        self, epoch: int, logs: Optional[Dict] = None, metric: Optional[float] = None
    ):
        pass

    def on_batch_begin(self, batch: int, logs: Optional[Dict] = None):
        pass

    def on_batch_end(self, batch: int, logs: Optional[Dict] = None):
        pass

    def on_train_begin(self, logs: Optional[Dict] = None):
        pass

    def on_train_end(self, logs: Optional[Dict] = None):
        pass

    def on_eval_begin(self, logs: Optional[Dict] = None):
        # at the moment only used to reset metrics before eval
        pass


class History(Callback):
    r"""Saves the metrics in the `history` attribute of the `Trainer`.

    TO DO: move this sentence to the docs, not here.
    This callback runs by default within `Trainer`, therefore, should not
    be passed to the `Trainer`. It is included here just for completion.
    """

    def on_train_begin(self, logs: Optional[Dict] = None):
        self.trainer.history = {}

    def on_epoch_end(
        self, epoch: int, logs: Optional[Dict] = None, metric: Optional[float] = None
    ):
        logs = logs or {}
        for k, v in logs.items():
            if isinstance(v, np.ndarray):
                v = v.tolist()
            if isinstance(v, list) and len(v) > 1:
                for i in range(len(v)):
                    self.trainer.history.setdefault(k + "_" + str(i), []).append(v[i])
            else:
                self.trainer.history.setdefault(k, []).append(v)


class LRShedulerCallback(Callback):
    r"""Callback for the learning rate schedulers to take a step

    TO DO: move this sentence to the docs, not here.
    This callback runs by default within `Trainer`, therefore, should not
    be passed to the `Trainer`. It is included here just for completion.
    """

    def on_batch_end(self, batch: int, logs: Optional[Dict] = None):
        if self.trainer.lr_scheduler is not None:
            if self._multiple_scheduler(self.trainer.lr_scheduler):
                for (
                    _,
                    scheduler,
                ) in self.trainer.lr_scheduler._schedulers.items():
                    if isinstance(scheduler, list):
                        for s in scheduler:
                            if self._is_cyclic(s):
                                s.step()
                    else:
                        if self._is_cyclic(scheduler):
                            scheduler.step()
            elif self.trainer.cyclic_lr:
                self.trainer.lr_scheduler.step()

    def on_epoch_end(  # noqa: C901
        self, epoch: int, logs: Optional[Dict] = None, metric: Optional[float] = None
    ):
        if self.trainer.lr_scheduler is not None:
            if self._multiple_scheduler(self.trainer.lr_scheduler):
                for (
                    _,
                    scheduler,
                ) in self.trainer.lr_scheduler._schedulers.items():
                    if isinstance(scheduler, list):
                        for s in scheduler:
                            if not self._is_cyclic(s):
                                if isinstance(s, ReduceLROnPlateau):
                                    s.step(metric)
                                else:
                                    s.step()
                    else:
                        if not self._is_cyclic(scheduler):
                            if isinstance(scheduler, ReduceLROnPlateau):
                                scheduler.step(metric)
                            else:
                                scheduler.step()
            elif not self.trainer.cyclic_lr:
                if isinstance(self.trainer.lr_scheduler, ReduceLROnPlateau):
                    self.trainer.lr_scheduler.step(metric)
                else:
                    self.trainer.lr_scheduler.step()

    @staticmethod
    def _multiple_scheduler(scheduler: LRScheduler) -> bool:
        return scheduler.__class__.__name__ == "MultipleLRScheduler"

    @staticmethod
    def _is_cyclic(scheduler: LRScheduler) -> bool:
        return "cycl" in scheduler.__class__.__name__.lower()


class MetricCallback(Callback):
    r"""Callback that resets the metrics (if any metric is used)

    This callback runs by default within `Trainer`, therefore, should not
    be passed to the `Trainer`. It is included here just for completion.
    """

    def __init__(self, container: MultipleMetrics):
        self.container = container

    def on_epoch_begin(self, epoch: int, logs: Optional[Dict] = None):
        self.container.reset()

    def on_eval_begin(self, logs: Optional[Dict] = None):
        self.container.reset()


class LRHistory(Callback):
    r"""Saves the learning rates during training in the `lr_history` attribute
    of the `Trainer`.

    Callbacks are passed as input parameters to the `Trainer` class. See
    `pytorch_widedeep.trainer.Trainer`

    Parameters
    ----------
    n_epochs: int
        number of training epochs

    Examples
    --------
    >>> from pytorch_widedeep.callbacks import LRHistory
    >>> from pytorch_widedeep.models import TabMlp, Wide, WideDeep
    >>> from pytorch_widedeep.training import Trainer
    >>>
    >>> embed_input = [(u, i, j) for u, i, j in zip(["a", "b", "c"][:4], [4] * 3, [8] * 3)]
    >>> column_idx = {k: v for v, k in enumerate(["a", "b", "c"])}
    >>> wide = Wide(10, 1)
    >>> deep = TabMlp(mlp_hidden_dims=[8, 4], column_idx=column_idx, cat_embed_input=embed_input)
    >>> model = WideDeep(wide, deep)
    >>> trainer = Trainer(model, objective="regression", callbacks=[LRHistory(n_epochs=10)])
    """

    def __init__(self, n_epochs: int):
        super(LRHistory, self).__init__()
        self.n_epochs = n_epochs

    def on_epoch_begin(self, epoch: int, logs: Optional[Dict] = None):
        if epoch == 0 and self.trainer.lr_scheduler is not None:
            self.trainer.lr_history = {}
            if self._multiple_scheduler(self.trainer.lr_scheduler):
                self._save_group_lr_mulitple_scheduler(step_location="on_epoch_begin")
            else:
                self._save_group_lr(self.trainer.optimizer)

    def on_batch_end(self, batch: int, logs: Optional[Dict] = None):
        if self.trainer.lr_scheduler is not None:
            if self._multiple_scheduler(self.trainer.lr_scheduler):
                self._save_group_lr_mulitple_scheduler(step_location="on_batch_end")
            elif self.trainer.cyclic_lr:
                self._save_group_lr(self.trainer.optimizer)

    def on_epoch_end(
        self, epoch: int, logs: Optional[Dict] = None, metric: Optional[float] = None
    ):
        if epoch != (self.n_epochs - 1) and self.trainer.lr_scheduler is not None:
            if self._multiple_scheduler(self.trainer.lr_scheduler):
                self._save_group_lr_mulitple_scheduler(step_location="on_epoch_end")
            elif not self.trainer.cyclic_lr:
                self._save_group_lr(self.trainer.optimizer)

    def _save_group_lr_mulitple_scheduler(self, step_location: str):
        for model_name, optimizer in self.trainer.optimizer._optimizers.items():
            if isinstance(optimizer, list):
                # then, if it has schedulers, we assume it has to have the
                # same number of schedulers as optimizers
                for i, opt in enumerate(optimizer):
                    if (
                        step_location == "on_epoch_begin"
                        or (
                            step_location == "on_batch_end"
                            and self._has_cyclic_scheduler(model_name)
                        )
                        or (
                            step_location == "on_epoch_end"
                            and not self._has_cyclic_scheduler(model_name)
                        )
                    ):
                        self._save_group_lr(opt, model_name, "_".join(["opt", str(i)]))
                else:
                    # do nothing
                    pass
            else:
                if (
                    step_location == "on_epoch_begin"
                    or (
                        step_location == "on_batch_end"
                        and self._has_cyclic_scheduler(model_name)
                    )
                    or (
                        step_location == "on_epoch_end"
                        and not self._has_cyclic_scheduler(model_name)
                    )
                ):
                    self._save_group_lr(optimizer, model_name)

    def _save_group_lr(
        self,
        opt: Optimizer,
        suffix: Optional[str] = None,
        model_name: Optional[str] = None,
    ):
        suffix = suffix or ""
        model_name = model_name or ""
        for group_idx, group in enumerate(opt.param_groups):
            group_name = ("_").join(
                [x for x in ["lr", model_name, suffix, str(group_idx)] if x]
            )
            self.trainer.lr_history.setdefault(group_name, []).append(group["lr"])

    @staticmethod
    def _multiple_scheduler(scheduler: LRScheduler) -> bool:
        return scheduler.__class__.__name__ == "MultipleLRScheduler"

    def _has_cyclic_scheduler(self, model_name: str):
        if model_name in self.trainer.lr_scheduler._schedulers:
            if isinstance(self.trainer.lr_scheduler._schedulers[model_name], list):
                return any(
                    [
                        self._is_cyclic(s)
                        for s in self.trainer.lr_scheduler._schedulers[model_name]
                    ]
                )
            else:
                return self._is_cyclic(
                    self.trainer.lr_scheduler._schedulers[model_name]
                )

    @staticmethod
    def _is_cyclic(scheduler: LRScheduler) -> bool:
        return "cycl" in scheduler.__class__.__name__.lower()


class ModelCheckpoint(Callback):
    r"""Saves the model after every epoch.

    This class is almost identical to the corresponding keras class.
    Therefore, **credit** to the Keras Team.

    Callbacks are passed as input parameters to the `Trainer` class. See
    `pytorch_widedeep.trainer.Trainer`

    Parameters
    ----------
    filepath: str, default=None
        Full path to save the output weights. It must contain only the root of
        the filenames. Epoch number and `.pt` extension (for pytorch) will be
        added. e.g. `filepath="path/to/output_weights/weights_out"` And the
        saved files in that directory will be named:
        _'weights_out_1.pt', 'weights_out_2.pt', ..._. If set to `None` the
        class just report best metric and best_epoch.
    monitor: str, default="loss"
        quantity to monitor. Typically _'val_loss'_ or metric name
        (e.g. _'val_acc'_)
    min_delta: float, default=0.
        minimum change in the monitored quantity to qualify as an
        improvement, i.e. an absolute change of less than min_delta, will
        count as no improvement.
    verbose:int, default=0
        verbosity mode
    save_best_only: bool, default=False,
        the latest best model according to the quantity monitored will not be
        overwritten.
    mode: str, default="auto"
        If `save_best_only=True`, the decision to overwrite the current save
        file is made based on either the maximization or the minimization of
        the monitored quantity. For _'acc'_, this should be _'max'_, for
        _'loss'_ this should be _'min'_, etc. In '_auto'_ mode, the
        direction is automatically inferred from the name of the monitored
        quantity.
    period: int, default=1
        Interval (number of epochs) between checkpoints.
    max_save: int, default=-1
        Maximum number of outputs to save. If -1 will save all outputs

    Attributes
    ----------
    best: float
        best metric
    best_epoch: int
        best epoch
    best_state_dict: dict
        best model state dictionary.<br/>
        To restore model to its best state use `Trainer.model.load_state_dict
        (model_checkpoint.best_state_dict)` where `model_checkpoint` is an
        instance of the class `ModelCheckpoint`. See the Examples folder in
        the repo or the Examples section in this documentation for details

    Examples
    --------
    >>> from pytorch_widedeep.callbacks import ModelCheckpoint
    >>> from pytorch_widedeep.models import TabMlp, Wide, WideDeep
    >>> from pytorch_widedeep.training import Trainer
    >>>
    >>> embed_input = [(u, i, j) for u, i, j in zip(["a", "b", "c"][:4], [4] * 3, [8] * 3)]
    >>> column_idx = {k: v for v, k in enumerate(["a", "b", "c"])}
    >>> wide = Wide(10, 1)
    >>> deep = TabMlp(mlp_hidden_dims=[8, 4], column_idx=column_idx, cat_embed_input=embed_input)
    >>> model = WideDeep(wide, deep)
    >>> trainer = Trainer(model, objective="regression", callbacks=[ModelCheckpoint(filepath='checkpoints/weights_out')])
    """

    def __init__(
        self,
        filepath: Optional[str] = None,
        monitor: str = "val_loss",
        min_delta: float = 0.0,
        verbose: int = 0,
        save_best_only: bool = False,
        mode: str = "auto",
        period: int = 1,
        max_save: int = -1,
    ):
        super(ModelCheckpoint, self).__init__()

        self.filepath = filepath
        self.monitor = monitor
        self.min_delta = min_delta
        self.verbose = verbose
        self.save_best_only = save_best_only
        self.mode = mode
        self.period = period
        self.max_save = max_save

        self.epochs_since_last_save = 0

        if self.filepath:
            if len(self.filepath.split("/")[:-1]) == 0:
                raise ValueError(
                    "'filepath' must be the full path to save the output weights,"
                    " including the root of the filenames. e.g. 'checkpoints/weights_out'"
                )

            root_dir = ("/").join(self.filepath.split("/")[:-1])
            if not os.path.exists(root_dir):
                os.makedirs(root_dir)

        if self.max_save > 0:
            self.old_files: List[str] = []

        if self.mode not in ["auto", "min", "max"]:
            warnings.warn(
                "ModelCheckpoint mode %s is unknown, "
                "fallback to auto mode." % (self.mode),
                RuntimeWarning,
            )
            self.mode = "auto"
        if self.mode == "min":
            self.monitor_op = np.less
            self.best = np.Inf
        elif self.mode == "max":
            self.monitor_op = np.greater  # type: ignore[assignment]
            self.best = -np.Inf
        else:
            if _is_metric(self.monitor):
                self.monitor_op = np.greater  # type: ignore[assignment]
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

        if self.monitor_op == np.greater:
            self.min_delta *= 1
        else:
            self.min_delta *= -1

    def on_epoch_end(  # noqa: C901
        self, epoch: int, logs: Optional[Dict] = None, metric: Optional[float] = None
    ):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            if self.filepath:
                filepath = "{}_{}.p".format(self.filepath, epoch + 1)
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn(
                        "Can save best model only with %s available, "
                        "skipping." % (self.monitor),
                        RuntimeWarning,
                    )
                else:
                    if self.monitor_op(current - self.min_delta, self.best):
                        if self.verbose > 0:
                            if self.filepath:
                                print(
                                    f"\nEpoch {epoch + 1}: {self.monitor} improved from {self.best:.5f} to {current:.5f} "
                                    f"Saving model to {filepath}"
                                )
                            else:
                                print(
                                    f"\nEpoch {epoch + 1}: {self.monitor} improved from {self.best:.5f} to {current:.5f} "
                                )
                        self.best = current
                        self.best_epoch = epoch
                        self.best_state_dict = copy.deepcopy(self.model.state_dict())
                        if self.filepath:
                            torch.save(self.best_state_dict, filepath)
                            if self.max_save > 0:
                                if len(self.old_files) == self.max_save:
                                    try:
                                        os.remove(self.old_files[0])
                                    except FileNotFoundError:
                                        pass
                                    self.old_files = self.old_files[1:]
                                self.old_files.append(filepath)
                    else:
                        if self.verbose > 0:
                            print(
                                f"\nEpoch {epoch + 1}: {self.monitor} did not improve from {self.best:.5f} "
                                f" considering a 'min_delta' improvement of {self.min_delta:.5f}"
                            )
            if not self.save_best_only and self.filepath:
                if self.verbose > 0:
                    print("\nEpoch %05d: saving model to %s" % (epoch + 1, filepath))
                torch.save(self.model.state_dict(), filepath)
                if self.max_save > 0:
                    if len(self.old_files) == self.max_save:
                        try:
                            os.remove(self.old_files[0])
                        except FileNotFoundError:
                            pass
                        self.old_files = self.old_files[1:]
                    self.old_files.append(filepath)

    def __getstate__(self):
        d = self.__dict__
        self_dict = {k: d[k] for k in d if k not in ["trainer", "model"]}
        return self_dict

    def __setstate__(self, state):
        self.__dict__ = state


class EarlyStopping(Callback):
    r"""Stop training when a monitored quantity has stopped improving.

    This class is almost identical to the corresponding keras class.
    Therefore, **credit** to the Keras Team.

    Callbacks are passed as input parameters to the `Trainer` class. See
    `pytorch_widedeep.trainer.Trainer`

    Parameters
    -----------
    monitor: str, default='val_loss'.
        Quantity to monitor. Typically _'val_loss'_ or metric name
        (e.g. _'val_acc'_)
    min_delta: float, default=0.
        minimum change in the monitored quantity to qualify as an
        improvement, i.e. an absolute change of less than min_delta, will
        count as no improvement.
    patience: int, default=10.
        Number of epochs that produced the monitored quantity with no
        improvement after which training will be stopped.
    verbose: int.
        verbosity mode.
    mode: str, default='auto'
        one of _{'auto', 'min', 'max'}_. In _'min'_ mode, training will
        stop when the quantity monitored has stopped decreasing; in _'max'_
        mode it will stop when the quantity monitored has stopped increasing;
        in _'auto'_ mode, the direction is automatically inferred from the
        name of the monitored quantity.
    baseline: float, Optional. default=None.
        Baseline value for the monitored quantity to reach. Training will
        stop if the model does not show improvement over the baseline.
    restore_best_weights: bool, default=None
        Whether to restore model weights from the epoch with the best
        value of the monitored quantity. If `False`, the model weights
        obtained at the last step of training are used.

    Attributes
    ----------
    best: float
        best metric
    stopped_epoch: int
        epoch when the training stopped

    Examples
    --------
    >>> from pytorch_widedeep.callbacks import EarlyStopping
    >>> from pytorch_widedeep.models import TabMlp, Wide, WideDeep
    >>> from pytorch_widedeep.training import Trainer
    >>>
    >>> embed_input = [(u, i, j) for u, i, j in zip(["a", "b", "c"][:4], [4] * 3, [8] * 3)]
    >>> column_idx = {k: v for v, k in enumerate(["a", "b", "c"])}
    >>> wide = Wide(10, 1)
    >>> deep = TabMlp(mlp_hidden_dims=[8, 4], column_idx=column_idx, cat_embed_input=embed_input)
    >>> model = WideDeep(wide, deep)
    >>> trainer = Trainer(model, objective="regression", callbacks=[EarlyStopping(patience=10)])
    """

    def __init__(
        self,
        monitor: str = "val_loss",
        min_delta: float = 0.0,
        patience: int = 10,
        verbose: int = 0,
        mode: str = "auto",
        baseline: Optional[float] = None,
        restore_best_weights: bool = False,
    ):
        super(EarlyStopping, self).__init__()

        self.monitor = monitor
        self.min_delta = min_delta
        self.patience = patience
        self.verbose = verbose
        self.mode = mode
        self.baseline = baseline
        self.restore_best_weights = restore_best_weights

        self.wait = 0
        self.stopped_epoch = 0
        self.state_dict = None

        if self.mode not in ["auto", "min", "max"]:
            warnings.warn(
                "EarlyStopping mode %s is unknown, "
                "fallback to auto mode." % self.mode,
                RuntimeWarning,
            )
            self.mode = "auto"

        if self.mode == "min":
            self.monitor_op = np.less
        elif self.mode == "max":
            self.monitor_op = np.greater  # type: ignore[assignment]
        else:
            if _is_metric(self.monitor):
                self.monitor_op = np.greater  # type: ignore[assignment]
            else:
                self.monitor_op = np.less

        if self.monitor_op == np.greater:
            self.min_delta *= 1
        else:
            self.min_delta *= -1

    def on_train_begin(self, logs: Optional[Dict] = None):
        # Allow instances to be re-used
        self.wait = 0
        self.stopped_epoch = 0
        if self.baseline is not None:
            self.best = self.baseline
        else:
            self.best = np.Inf if self.monitor_op == np.less else -np.Inf

    def on_epoch_end(
        self, epoch: int, logs: Optional[Dict] = None, metric: Optional[float] = None
    ):
        current = self.get_monitor_value(logs)
        if current is None:
            return

        if self.monitor_op(current - self.min_delta, self.best):
            self.best = current
            self.wait = 0
            self.best_epoch = epoch
            if self.restore_best_weights:
                self.state_dict = copy.deepcopy(self.model.state_dict())
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.trainer.early_stop = True

    def on_train_end(self, logs: Optional[Dict] = None):
        if self.stopped_epoch > 0 and self.verbose > 0:
            print(
                f"Best Epoch: {self.best_epoch + 1}. Best {self.monitor}: {self.best:.5f}"
            )
        if self.restore_best_weights and self.state_dict is not None:
            if self.verbose > 0:
                print("Restoring model weights from the end of the best epoch")
            self.model.load_state_dict(self.state_dict)

    def get_monitor_value(self, logs):
        monitor_value = logs.get(self.monitor)
        if monitor_value is None:
            warnings.warn(
                "Early stopping conditioned on metric `%s` "
                "which is not available. Available metrics are: %s"
                % (self.monitor, ",".join(list(logs.keys()))),
                RuntimeWarning,
            )
        return monitor_value

    def __getstate__(self):
        d = self.__dict__
        self_dict = {k: d[k] for k in d if k not in ["trainer", "model"]}
        return self_dict

    def __setstate__(self, state):
        self.__dict__ = state
