import os
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import trange
from torchmetrics import Metric as TorchMetric
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau

from pytorch_widedeep.metrics import Metric, MultipleMetrics
from pytorch_widedeep.wdtypes import *  # noqa: F403
from pytorch_widedeep.callbacks import (
    History,
    Callback,
    MetricCallback,
    CallbackContainer,
    LRShedulerCallback,
)
from pytorch_widedeep.utils.general_utils import Alias
from pytorch_widedeep.training._trainer_utils import (
    save_epoch_logs,
    print_loss_and_metric,
    bayesian_alias_to_loss,
    tabular_train_val_split,
)
from pytorch_widedeep.bayesian_models._base_bayesian_model import (
    BaseBayesianModel,
)


class BayesianTrainer:
    r"""Class to set the of attributes that will be used during the
    training process.

    Parameters
    ----------
    model: ``BaseBayesianModel``
        An object of class ``BaseBayesianModel``
    objective: str
        Defines the objective, loss or cost function.

        Param aliases: ``loss_function``, ``loss_fn``, ``loss``,
        ``cost_function``, ``cost_fn``, ``cost``

        Possible values are: 'binary', 'multiclass', 'regression'

    custom_loss_function: ``nn.Module``, optional, default = None
        object of class ``nn.Module``. If none of the loss functions
        available suits the user, it is possible to pass a custom loss
        function. See for example
        :class:`pytorch_widedeep.losses.FocalLoss` for the required
        structure of the object or the `Examples
        <https://github.com/jrzaurin/pytorch-widedeep/tree/master/examples>`__
        folder in the repo.
    optimizer: ``Optimzer``, optional, default= None
        An instance of Pytorch's ``Optimizer`` object
        (e.g. :obj:`torch.optim.Adam()`). if no optimizer is passed it will
        default to ``AdamW``.
    lr_schedulers: ``LRScheduler``, optional, default=None
        An instance of Pytorch's ``LRScheduler`` object (e.g
        :obj:`torch.optim.lr_scheduler.StepLR(opt, step_size=5)`)
    callbacks: List, optional, default=None
        List with :obj:`Callback` objects. The three callbacks available in
        ``pytorch-widedeep`` are: ``LRHistory``, ``ModelCheckpoint`` and
        ``EarlyStopping``. The ``History`` and the ``LRShedulerCallback``
        callbacks are used by default. This can also be a custom callback as
        long as the object of type ``Callback``. See
        :obj:`pytorch_widedeep.callbacks.Callback` or the `Examples
        <https://github.com/jrzaurin/pytorch-widedeep/tree/master/examples>`__
        folder in the repo
    metrics: List, optional, default=None
        - List of objects of type :obj:`Metric`. Metrics available are:
          ``Accuracy``, ``Precision``, ``Recall``, ``FBetaScore``,
          ``F1Score`` and ``R2Score``. This can also be a custom metric as
          long as it is an object of type :obj:`Metric`. See
          :obj:`pytorch_widedeep.metrics.Metric` or the `Examples
          <https://github.com/jrzaurin/pytorch-widedeep/tree/master/examples>`__
          folder in the repo
        - List of objects of type :obj:`torchmetrics.Metric`. This can be any
          metric from torchmetrics library `Examples
          <https://torchmetrics.readthedocs.io/en/latest/references/modules.html#
          classification-metrics>`_. This can also be a custom metric as
          long as it is an object of type :obj:`Metric`. See `the instructions
          <https://torchmetrics.readthedocs.io/en/latest/>`_.
    verbose: int, default=1
        Setting it to 0 will print nothing during training.
    seed: int, default=1
        Random seed to be used internally for train_test_split

    Attributes
    ----------
    cyclic_lr: bool
        Attribute that indicates if  the lr_scheduler is cyclic_lr
        (i.e. ``CyclicLR`` or ``OneCycleLR``). See `Pytorch schedulers
        <https://pytorch.org/docs/stable/optim.html>`_.
    """

    @Alias(  # noqa: C901
        "objective",
        ["loss_function", "loss_fn", "loss", "cost_function", "cost_fn", "cost"],
    )
    def __init__(
        self,
        model: BaseBayesianModel,
        objective: str,
        custom_loss_function: Optional[Module] = None,
        optimizer: Optimizer = None,
        lr_scheduler: LRScheduler = None,
        callbacks: Optional[List[Callback]] = None,
        metrics: Optional[Union[List[Metric], List[TorchMetric]]] = None,
        verbose: int = 1,
        seed: int = 1,
        **kwargs,
    ):

        if objective not in ["binary", "multiclass", "regression"]:
            raise ValueError(
                "If 'custom_loss_function' is not None, 'objective' must be 'binary' "
                "'multiclass' or 'regression', consistent with the loss function"
            )

        self.device, self.num_workers = self._set_device_and_num_workers(**kwargs)

        self.model = model
        self.early_stop = False

        self.verbose = verbose
        self.seed = seed
        self.objective = objective

        self.loss_fn = self._set_loss_fn(objective, custom_loss_function, **kwargs)
        self.optimizer = (
            optimizer
            if optimizer is not None
            else torch.optim.AdamW(self.model.parameters())
        )
        self.lr_scheduler = lr_scheduler
        self._set_lr_scheduler_running_params(lr_scheduler, **kwargs)
        self._set_callbacks_and_metrics(callbacks, metrics)
        self.model.to(self.device)

    def fit(  # noqa: C901
        self,
        X_tab: np.ndarray,
        target: np.ndarray,
        X_tab_val: Optional[np.ndarray] = None,
        target_val: Optional[np.ndarray] = None,
        val_split: Optional[float] = None,
        n_epochs: int = 1,
        val_freq: int = 1,
        batch_size: int = 32,
        n_train_samples: int = 2,
        n_val_samples: int = 2,
    ):
        r"""Fit method.

        Parameters
        ----------
        X_tab: np.ndarray,
            tabular dataset
        target: np.ndarray
            target values
        X_tab_val: np.ndarray, Optional, default = None
            validation data
        target_val: np.ndarray, Optional, default = None
            validation target values
        val_split: float, Optional. default=None
            An alterative to passing the validation set is to use a train/val
            split fraction via 'val_split'
        n_epochs: int, default=1
            number of epochs
        validation_freq: int, default=1
            epochs validation frequency
        batch_size: int, default=32
            batch size
        n_train_samples: int, default=2
            number of samples to average over during the training process.
        n_val_samples: int, default=2
            number of samples to average over during the validation process.
        """

        self.batch_size = batch_size

        train_set, eval_set = tabular_train_val_split(
            self.seed, self.objective, X_tab, target, X_tab_val, target_val, val_split
        )
        train_loader = DataLoader(
            dataset=train_set, batch_size=batch_size, num_workers=self.num_workers
        )
        train_steps = len(train_loader)

        if eval_set is not None:
            eval_loader = DataLoader(
                dataset=eval_set,
                batch_size=batch_size,
                num_workers=self.num_workers,
                shuffle=False,
            )
            eval_steps = len(eval_loader)

        self.callback_container.on_train_begin(
            {
                "batch_size": batch_size,
                "train_steps": train_steps,
                "n_epochs": n_epochs,
            }
        )
        for epoch in range(n_epochs):
            epoch_logs: Dict[str, float] = {}
            self.callback_container.on_epoch_begin(epoch, logs=epoch_logs)

            self.train_running_loss = 0.0
            with trange(train_steps, disable=self.verbose != 1) as t:
                for batch_idx, (X, y) in zip(t, train_loader):
                    t.set_description("epoch %i" % (epoch + 1))
                    train_score, train_loss = self._train_step(
                        X, y, n_train_samples, train_steps, batch_idx
                    )
                    print_loss_and_metric(t, train_loss, train_score)
                    self.callback_container.on_batch_end(batch=batch_idx)
            epoch_logs = save_epoch_logs(epoch_logs, train_loss, train_score, "train")

            on_epoch_end_metric = None
            if eval_set is not None and epoch % val_freq == (val_freq - 1):
                self.callback_container.on_eval_begin()
                self.valid_running_loss = 0.0
                with trange(eval_steps, disable=self.verbose != 1) as v:
                    for i, (X, y) in zip(v, eval_loader):
                        v.set_description("valid")
                        val_score, val_loss = self._eval_step(
                            X, y, n_val_samples, train_steps, i
                        )
                        print_loss_and_metric(v, val_loss, val_score)
                epoch_logs = save_epoch_logs(epoch_logs, val_loss, val_score, "val")

                if self.reducelronplateau:
                    if self.reducelronplateau_criterion == "loss":
                        on_epoch_end_metric = val_loss
                    else:
                        on_epoch_end_metric = val_score[
                            self.reducelronplateau_criterion
                        ]

            self.callback_container.on_epoch_end(epoch, epoch_logs, on_epoch_end_metric)

            if self.early_stop:
                self.callback_container.on_train_end(epoch_logs)
                break

        self.callback_container.on_train_end(epoch_logs)
        self._restore_best_weights()
        self.model.train()

    def predict(  # type: ignore[return]
        self,
        X_tab: np.ndarray,
        n_samples: int = 5,
        return_samples: bool = False,
        batch_size: int = 256,
    ) -> np.ndarray:
        r"""Returns the predictions

        Parameters
        ----------
        X_tab: np.ndarray,
            tabular dataset
        n_samples: int, default=5
            number of samples that will be either returned or averaged to
            produce an overal prediction
        return_samples: bool, default = False
            Boolean indicating whether the n samples will be averaged or directly returned
        batch_size: int, default = 256
            batch size
        """

        preds_l = self._predict(X_tab, n_samples, return_samples, batch_size)
        preds = np.hstack(preds_l) if return_samples else np.vstack(preds_l)
        axis = 2 if return_samples else 1

        if self.objective == "regression":
            return preds.squeeze(axis)
        if self.objective == "binary":
            return (preds.squeeze(axis) > 0.5).astype("int")
        if self.objective == "multiclass":
            return np.argmax(preds, axis)

    def predict_proba(  # type: ignore[return]
        self,
        X_tab: np.ndarray,
        n_samples: int = 5,
        return_samples: bool = False,
        batch_size: int = 256,
    ) -> np.ndarray:
        r"""Returns the predicted probabilities

        Parameters
        ----------
        X_tab: np.ndarray,
            tabular dataset
        n_samples: int, default=5
            number of samples that will be either returned or averaged to
            produce an overal prediction
        return_samples: bool, default = False
            Boolean indicating whether the n samples will be averaged or directly returned
        batch_size: int, default = 256
            batch size
        """
        preds_l = self._predict(X_tab, n_samples, return_samples, batch_size)
        preds = np.hstack(preds_l) if return_samples else np.vstack(preds_l)

        if self.objective == "binary":
            if return_samples:
                preds = preds.squeeze(2)
                probs = np.zeros([n_samples, preds.shape[1], 2])
                for i in range(n_samples):
                    probs[i, :, 0] = 1 - preds[i]
                    probs[i, :, 1] = preds[i]
            else:
                preds = preds.squeeze(1)
                probs = np.zeros([preds.shape[0], 2])
                probs[:, 0] = 1 - preds
                probs[:, 1] = preds
            return probs
        if self.objective == "multiclass":
            return preds

    def save(
        self,
        path: str,
        save_state_dict: bool = False,
        model_filename: str = "wd_model.pt",
    ):
        r"""Saves the model, training and evaluation history, and the
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
        save_state_dict: bool, default = False
            Boolean indicating whether to save directly the model or the
            model's state dictionary
        model_filename: str, Optional, default = "wd_model.pt"
            filename where the model weights will be store
        """

        save_dir = Path(path)
        history_dir = save_dir / "history"
        history_dir.mkdir(exist_ok=True, parents=True)

        # the trainer is run with the History Callback by default
        with open(history_dir / "train_eval_history.json", "w") as teh:
            json.dump(self.history, teh)  # type: ignore[attr-defined]

        has_lr_history = any(
            [clbk.__class__.__name__ == "LRHistory" for clbk in self.callbacks]
        )
        if self.lr_scheduler is not None and has_lr_history:
            with open(history_dir / "lr_history.json", "w") as lrh:
                json.dump(self.lr_history, lrh)  # type: ignore[attr-defined]

        model_path = save_dir / model_filename
        if save_state_dict:
            torch.save(self.model.state_dict(), model_path)
        else:
            torch.save(self.model, model_path)

    def _restore_best_weights(self):
        already_restored = any(
            [
                (
                    callback.__class__.__name__ == "EarlyStopping"
                    and callback.restore_best_weights
                )
                for callback in self.callback_container.callbacks
            ]
        )
        if already_restored:
            pass
        else:
            for callback in self.callback_container.callbacks:
                if callback.__class__.__name__ == "ModelCheckpoint":
                    if callback.save_best_only:
                        if self.verbose:
                            print(
                                f"Model weights restored to best epoch: {callback.best_epoch + 1}"
                            )
                        self.model.load_state_dict(callback.best_state_dict)
                    else:
                        if self.verbose:
                            print(
                                "Model weights after training corresponds to the those of the "
                                "final epoch which might not be the best performing weights. Use"
                                "the 'ModelCheckpoint' Callback to restore the best epoch weights."
                            )

    def _train_step(
        self,
        X_tab: Tensor,
        target: Tensor,
        n_samples: int,
        n_batches: int,
        batch_idx: int,
    ):

        self.model.train()

        X = X_tab.to(self.device)
        y = target.view(-1, 1).float() if self.objective != "multiclass" else target
        y = y.to(self.device)

        self.optimizer.zero_grad()
        y_pred, loss = self.model.sample_elbo(X, y, self.loss_fn, n_samples, n_batches)  # type: ignore[arg-type]

        y_pred = y_pred.mean(dim=0)
        score = self._get_score(y_pred, y)

        loss.backward()
        self.optimizer.step()

        self.train_running_loss += loss.item()
        avg_loss = self.train_running_loss / (batch_idx + 1)

        return score, avg_loss

    def _eval_step(
        self,
        X_tab: Tensor,
        target: Tensor,
        n_samples: int,
        n_batches: int,
        batch_idx: int,
    ):

        self.model.eval()
        with torch.no_grad():
            X = X_tab.to(self.device)
            y = target.view(-1, 1).float() if self.objective != "multiclass" else target
            y = y.to(self.device)

            y_pred, loss = self.model.sample_elbo(
                X,  # type: ignore[arg-type]
                y,
                self.loss_fn,
                n_samples,
                n_batches,
            )
            y_pred = y_pred.mean(dim=0)
            score = self._get_score(y_pred, y)

            self.valid_running_loss += loss.item()
            avg_loss = self.valid_running_loss / (batch_idx + 1)

        return score, avg_loss

    def _get_score(self, y_pred, y):
        if self.metric is not None:
            if self.objective == "regression":
                score = self.metric(y_pred, y)
            if self.objective == "binary":
                score = self.metric(torch.sigmoid(y_pred), y)
            if self.objective == "multiclass":
                score = self.metric(F.softmax(y_pred, dim=1), y)
            return score
        else:
            return None

    def _predict(  # noqa: C901
        self,
        X_tab: np.ndarray = None,
        n_samples: int = 5,
        return_samples: bool = False,
        batch_size: int = 256,
    ) -> List:

        self.batch_size = batch_size

        test_set = TensorDataset(torch.from_numpy(X_tab))
        test_loader = DataLoader(
            dataset=test_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )
        test_steps = (len(test_loader.dataset) // test_loader.batch_size) + 1  # type: ignore[arg-type]

        preds_l = []
        with torch.no_grad():
            with trange(test_steps, disable=self.verbose != 1) as tt:
                for j, Xl in zip(tt, test_loader):
                    tt.set_description("predict")

                    X = Xl[0].to(self.device)

                    if return_samples:
                        preds = torch.stack([self.model(X) for _ in range(n_samples)])
                    else:
                        self.model.eval()
                        preds = self.model(X)

                    if self.objective == "binary":
                        preds = torch.sigmoid(preds)
                    if self.objective == "multiclass":
                        preds = (
                            F.softmax(preds, dim=2)
                            if return_samples
                            else F.softmax(preds, dim=1)
                        )

                    preds = preds.cpu().data.numpy()
                    preds_l.append(preds)

        self.model.train()

        return preds_l

    def _set_loss_fn(self, objective, custom_loss_function, **kwargs):

        if custom_loss_function is not None:
            return custom_loss_function

        class_weight = (
            torch.tensor(kwargs["class_weight"]).to(self.device)
            if "class_weight" in kwargs
            else None
        )

        if self.objective != "regression":
            return bayesian_alias_to_loss(objective, weight=class_weight)
        else:
            return bayesian_alias_to_loss(objective)

    def _set_reduce_on_plateau_criterion(
        self, lr_scheduler, reducelronplateau_criterion
    ):

        self.reducelronplateau = False

        if isinstance(lr_scheduler, ReduceLROnPlateau):
            self.reducelronplateau = True

        if self.reducelronplateau and not reducelronplateau_criterion:
            UserWarning(
                "The learning rate scheduler is of type ReduceLROnPlateau. The step method in this"
                " scheduler requires a 'metrics' param that can be either the validation loss or the"
                " validation metric. Please, when instantiating the Trainer, specify which quantity"
                " will be tracked using reducelronplateau_criterion = 'loss' (default) or"
                " reducelronplateau_criterion = 'metric'"
            )
        else:
            self.reducelronplateau_criterion = "loss"

    def _set_lr_scheduler_running_params(self, lr_scheduler, **kwargs):
        # ReduceLROnPlateau is special

        reducelronplateau_criterion = kwargs.get("reducelronplateau_criterion", None)
        self._set_reduce_on_plateau_criterion(lr_scheduler, reducelronplateau_criterion)
        if lr_scheduler is not None:
            self.cyclic_lr = "cycl" in lr_scheduler.__class__.__name__.lower()
        else:
            self.cyclic_lr = False

    def _set_callbacks_and_metrics(self, callbacks, metrics):
        self.callbacks: List = [History(), LRShedulerCallback()]
        if callbacks is not None:
            for callback in callbacks:
                if isinstance(callback, type):
                    callback = callback()
                self.callbacks.append(callback)
        if metrics is not None:
            self.metric = MultipleMetrics(metrics)
            self.callbacks += [MetricCallback(self.metric)]
        else:
            self.metric = None
        self.callback_container = CallbackContainer(self.callbacks)
        self.callback_container.set_model(self.model)
        self.callback_container.set_trainer(self)

    @staticmethod
    def _set_device_and_num_workers(**kwargs):

        # Important note for Mac users: Since python 3.8, the multiprocessing
        # library start method changed from 'fork' to 'spawn'. This affects the
        # data-loaders, which will not run in parallel.
        default_num_workers = (
            0
            if sys.platform == "darwin" and sys.version_info.minor > 7
            else os.cpu_count()
        )
        default_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device = kwargs.get("device", default_device)
        num_workers = kwargs.get("num_workers", default_num_workers)
        return device, num_workers
