"""
Code here is mostly based on the code from the torchsample and Keras packages

CREDIT TO THE TORCHSAMPLE AND KERAS TEAMS
"""
import os
import datetime
import warnings

import numpy as np
import torch

from pytorch_widedeep.wdtypes import *  # noqa: F403


def _get_current_time():
    return datetime.datetime.now().strftime("%B %d, %Y - %I:%M%p")


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

    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None):
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_epoch_end(epoch, logs)

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

    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None):
        pass

    def on_batch_begin(self, batch: int, logs: Optional[Dict] = None):
        pass

    def on_batch_end(self, batch: int, logs: Optional[Dict] = None):
        pass

    def on_train_begin(self, logs: Optional[Dict] = None):
        pass

    def on_train_end(self, logs: Optional[Dict] = None):
        pass


class History(Callback):
    r"""Callback that records metrics to a ``history`` attribute.

    This callback runs by default within :obj:`Trainer`, therefore, should not
    be passed to the ``Trainer``. Is included here just for completion.
    """

    def on_train_begin(self, logs: Optional[Dict] = None):
        self.trainer.history = {}

    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None):
        logs = logs or {}
        for k, v in logs.items():
            self.trainer.history.setdefault(k, []).append(v)


class LRHistory(Callback):
    def __init__(self, n_epochs: int):
        r"""Saves the learning rates during training to a ``lr_history`` attribute.

        Callbacks are passed as input parameters to the ``Trainer`` class. See
        :class:`pytorch_widedeep.trainer.Trainer`

        Parameters
        ----------
        n_epochs: int
            number of epochs durint training

        Examples
        --------
        >>> from pytorch_widedeep.callbacks import LRHistory
        >>> from pytorch_widedeep.models import TabMlp, Wide, WideDeep
        >>> from pytorch_widedeep.training import Trainer
        >>>
        >>> embed_input = [(u, i, j) for u, i, j in zip(["a", "b", "c"][:4], [4] * 3, [8] * 3)]
        >>> column_idx = {k: v for v, k in enumerate(["a", "b", "c"])}
        >>> wide = Wide(10, 1)
        >>> deep = TabMlp(mlp_hidden_dims=[8, 4], column_idx=column_idx, embed_input=embed_input)
        >>> model = WideDeep(wide, deep)
        >>> trainer = Trainer(model, objective="regression", callbacks=[LRHistory(n_epochs=10)])
        """
        super(LRHistory, self).__init__()
        self.n_epochs = n_epochs

    def on_epoch_begin(self, epoch: int, logs: Optional[Dict] = None):
        if epoch == 0 and self.trainer.lr_scheduler is not None:
            self.trainer.lr_history = {}
            if self._multiple_scheduler():
                self._save_group_lr_mulitple_scheduler(step_location="on_epoch_begin")
            else:
                self._save_group_lr(self.trainer.optimizer)

    def on_batch_end(self, batch: int, logs: Optional[Dict] = None):
        if self.trainer.lr_scheduler is not None:
            if self._multiple_scheduler():
                self._save_group_lr_mulitple_scheduler(step_location="on_batch_end")
            elif self.trainer.cyclic_lr:
                self._save_group_lr(self.trainer.optimizer)

    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None):
        if epoch != (self.n_epochs - 1) and self.trainer.lr_scheduler is not None:
            if self._multiple_scheduler():
                self._save_group_lr_mulitple_scheduler(step_location="on_epoch_end")
            elif not self.trainer.cyclic_lr:
                self._save_group_lr(self.trainer.optimizer)

    def _save_group_lr_mulitple_scheduler(self, step_location: str):
        for model_name, opt in self.trainer.optimizer._optimizers.items():
            if step_location == "on_epoch_begin":
                self._save_group_lr(opt, model_name)
            if step_location == "on_batch_end":
                if self._is_cyclic(model_name):
                    self._save_group_lr(opt, model_name)
            if step_location == "on_epoch_end":
                if not self._is_cyclic(model_name):
                    self._save_group_lr(opt, model_name)

    def _save_group_lr(self, opt: Optimizer, model_name: Optional[str] = None):
        for group_idx, group in enumerate(opt.param_groups):
            if model_name is not None:
                group_name = ("_").join(["lr", model_name, str(group_idx)])
            else:
                group_name = ("_").join(["lr", str(group_idx)])
            self.trainer.lr_history.setdefault(group_name, []).append(group["lr"])

    def _multiple_scheduler(self):
        return self.trainer.lr_scheduler.__class__.__name__ == "MultipleLRScheduler"

    def _is_cyclic(self, model_name: str):
        return (
            self._has_scheduler(model_name)
            and "cycl"
            in self.trainer.lr_scheduler._schedulers[
                model_name
            ].__class__.__name__.lower()
        )

    def _has_scheduler(self, model_name: str):
        return model_name in self.trainer.lr_scheduler._schedulers


class ModelCheckpoint(Callback):
    def __init__(
        self,
        filepath: str,
        monitor: str = "val_loss",
        verbose: int = 0,
        save_best_only: bool = False,
        mode: str = "auto",
        period: int = 1,
        max_save: int = -1,
    ):
        r"""Saves the model after every epoch.

        This class is almost identical to the corresponding keras class.
        Therefore, **credit** to the Keras Team.

        Callbacks are passed as input parameters to the ``Trainer`` class. See
        :class:`pytorch_widedeep.trainer.Trainer`

        Parameters
        ----------
        filepath: str
            Full path to save the output weights. It must contain only the root of
            the filenames. Epoch number and ``.pt`` extension (for pytorch) will
            be added. e.g. ``filepath="path/to/output_weights/weights_out"`` And
            the saved files in that directory will be named: ``weights_out_1.pt,
            weights_out_2.pt, ...``
        monitor: str, default="val_loss"
            quantity to monitor. :obj:`ModelCheckpoint` will infer if this is a
            loss (i.e. contains the str `'loss'`) or a metric (i.e. contains the
            str `'acc'` or starts with `'fmeasure'`).
        verbose:int, default=0,
            verbosity mode
        save_best_only: bool, default=False,
            the latest best model according to the quantity monitored will not be
            overwritten.
        mode: str, default="auto",
            If ``save_best_only=True``, the decision to overwrite the current save
            file is made based on either the maximization or the minimization of
            the monitored quantity. For `'val_acc'`, this should be `'max'`, for
            `'val_loss'` this should be `'min'`, etc. In `'auto'` mode, the
            direction is automatically inferred from the name of the monitored
            quantity.
        period: int, default=1,
            Interval (number of epochs) between checkpoints.
        max_save: int, default=-1
            Maximum number of outputs to save. If -1 will save all outputs

        Examples
        --------
        >>> from pytorch_widedeep.callbacks import ModelCheckpoint
        >>> from pytorch_widedeep.models import TabMlp, Wide, WideDeep
        >>> from pytorch_widedeep.training import Trainer
        >>>
        >>> embed_input = [(u, i, j) for u, i, j in zip(["a", "b", "c"][:4], [4] * 3, [8] * 3)]
        >>> column_idx = {k: v for v, k in enumerate(["a", "b", "c"])}
        >>> wide = Wide(10, 1)
        >>> deep = TabMlp(mlp_hidden_dims=[8, 4], column_idx=column_idx, embed_input=embed_input)
        >>> model = WideDeep(wide, deep)
        >>> trainer = Trainer(model, objective="regression", callbacks=[ModelCheckpoint(filepath='checkpoints/weights_out')])
        """
        super(ModelCheckpoint, self).__init__()

        self.filepath = filepath
        self.monitor = monitor
        self.verbose = verbose
        self.save_best_only = save_best_only
        self.mode = mode
        self.period = period
        self.max_save = max_save

        self.epochs_since_last_save = 0

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
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if "acc" in self.monitor or self.monitor.startswith("fmeasure"):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None):  # noqa: C901
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
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
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print(
                                "\nEpoch %05d: %s improved from %0.5f to %0.5f,"
                                " saving model to %s"
                                % (
                                    epoch + 1,
                                    self.monitor,
                                    self.best,
                                    current,
                                    filepath,
                                )
                            )
                        self.best = current
                        torch.save(self.model.state_dict(), filepath)
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
                                "\nEpoch %05d: %s did not improve from %0.5f"
                                % (epoch + 1, self.monitor, self.best)
                            )
            else:
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


class EarlyStopping(Callback):
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
        r"""Stop training when a monitored quantity has stopped improving.

        This class is almost identical to the corresponding keras class.
        Therefore, **credit** to the Keras Team.

        Callbacks are passed as input parameters to the ``Trainer`` class. See
        :class:`pytorch_widedeep.trainer.Trainer`

        Parameters
        -----------
        monitor: str, default='val_loss'.
            Quantity to be monitored.
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
            one of {'`auto`', '`min`', '`max`'}. In `'min'` mode, training will
            stop when the quantity monitored has stopped decreasing; in `'max'`
            mode it will stop when the quantity monitored has stopped increasing;
            in `'auto'` mode, the direction is automatically inferred from the
            name of the monitored quantity.
        baseline: float, Optional. default=None.
            Baseline value for the monitored quantity to reach. Training will
            stop if the model does not show improvement over the baseline.
        restore_best_weights: bool, default=None
            Whether to restore model weights from the epoch with the best
            value of the monitored quantity. If ``False``, the model weights
            obtained at the last step of training are used.

        Examples
        --------
        >>> from pytorch_widedeep.callbacks import EarlyStopping
        >>> from pytorch_widedeep.models import TabMlp, Wide, WideDeep
        >>> from pytorch_widedeep.training import Trainer
        >>>
        >>> embed_input = [(u, i, j) for u, i, j in zip(["a", "b", "c"][:4], [4] * 3, [8] * 3)]
        >>> column_idx = {k: v for v, k in enumerate(["a", "b", "c"])}
        >>> wide = Wide(10, 1)
        >>> deep = TabMlp(mlp_hidden_dims=[8, 4], column_idx=column_idx, embed_input=embed_input)
        >>> model = WideDeep(wide, deep)
        >>> trainer = Trainer(model, objective="regression", callbacks=[EarlyStopping(patience=10)])
        """
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
            self.monitor_op = np.greater
        else:
            if "acc" in self.monitor:
                self.monitor_op = np.greater
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

    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None):
        current = self.get_monitor_value(logs)
        if current is None:
            return

        if self.monitor_op(current - self.min_delta, self.best):
            self.best = current
            self.wait = 0
            if self.restore_best_weights:
                self.state_dict = self.model.state_dict()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.trainer.early_stop = True
                if self.restore_best_weights:
                    if self.verbose > 0:
                        print(
                            "Restoring model weights from the end of " "the best epoch"
                        )
                    self.model.load_state_dict(self.state_dict)

    def on_train_end(self, logs: Optional[Dict] = None):
        if self.stopped_epoch > 0 and self.verbose > 0:
            print("Epoch %05d: early stopping" % (self.stopped_epoch + 1))

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
