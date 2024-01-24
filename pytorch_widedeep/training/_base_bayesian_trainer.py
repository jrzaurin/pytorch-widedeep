import os
import sys
import warnings
from abc import ABC, abstractmethod

import numpy as np
import torch
from torchmetrics import Metric as TorchMetric
from torch.optim.lr_scheduler import ReduceLROnPlateau

from pytorch_widedeep.metrics import Metric, MultipleMetrics
from pytorch_widedeep.wdtypes import (
    Any,
    List,
    Union,
    Module,
    Optional,
    Optimizer,
    LRScheduler,
)
from pytorch_widedeep.callbacks import (
    History,
    Callback,
    MetricCallback,
    CallbackContainer,
    LRShedulerCallback,
)
from pytorch_widedeep.training._trainer_utils import bayesian_alias_to_loss
from pytorch_widedeep.bayesian_models._base_bayesian_model import (
    BaseBayesianModel,
)


# There are some nuances in the Bayesian Trainer that make it hard to build an
# overall BaseTrainer. We could still perhaps code a very basic Trainer and
# then pass it to a BaseTrainer and BaseBayesianTrainer. However in this
# particular case we prefer code repetition as we believe is a simpler
# solution
class BaseBayesianTrainer(ABC):
    def __init__(
        self,
        model: BaseBayesianModel,
        objective: str,
        custom_loss_function: Optional[Module],
        optimizer: Optional[Optimizer],
        lr_scheduler: Optional[LRScheduler],
        callbacks: Optional[List[Callback]],
        metrics: Optional[Union[List[Metric], List[TorchMetric]]],
        verbose: int,
        seed: int,
        **kwargs,
    ):
        if objective not in ["binary", "multiclass", "regression"]:
            raise ValueError(
                "If 'custom_loss_function' is not None, 'objective' must be 'binary' "
                "'multiclass' or 'regression', consistent with the loss function"
            )

        self.device, self.num_workers = self._set_device_and_num_workers(**kwargs)

        self.early_stop = False
        self.model = model
        self.model.to(self.device)

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

    @abstractmethod
    def fit(
        self,
        X_tab: np.ndarray,
        target: np.ndarray,
        X_tab_val: Optional[np.ndarray],
        target_val: Optional[np.ndarray],
        val_split: Optional[float],
        n_epochs: int,
        validation_freq: int,
        batch_size: int,
        n_train_samples: int,
        n_val_samples: int,
    ):
        raise NotImplementedError("Trainer.fit method not implemented")

    @abstractmethod
    def predict(
        self,
        X_tab: np.ndarray,
        n_samples: int,
        return_samples: bool,
        batch_size: int,
    ) -> np.ndarray:
        raise NotImplementedError("Trainer.predict method not implemented")

    @abstractmethod
    def predict_proba(
        self,
        X_tab: np.ndarray,
        n_samples: int,
        return_samples: bool,
        batch_size: int,
    ) -> np.ndarray:
        raise NotImplementedError("Trainer.predict_proba method not implemented")

    @abstractmethod
    def save(
        self,
        path: str,
        save_state_dict: bool,
        model_filename: str,
    ):
        raise NotImplementedError("Trainer.save method not implemented")

    def _restore_best_weights(self):  # noqa: C901
        early_stopping_min_delta = None
        model_checkpoint_min_delta = None
        already_restored = False

        for callback in self.callback_container.callbacks:
            if (
                callback.__class__.__name__ == "EarlyStopping"
                and callback.restore_best_weights
            ):
                early_stopping_min_delta = callback.min_delta
                already_restored = True

            if callback.__class__.__name__ == "ModelCheckpoint":
                model_checkpoint_min_delta = callback.min_delta

        if (
            early_stopping_min_delta is not None
            and model_checkpoint_min_delta is not None
        ) and (early_stopping_min_delta != model_checkpoint_min_delta):
            warnings.warn(
                "'min_delta' is different in the 'EarlyStopping' and 'ModelCheckpoint' callbacks. "
                "This implies a different definition of 'improvement' for these two callbacks",
                UserWarning,
            )

        if already_restored:
            # already restored via EarlyStopping
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
                                "final epoch which might not be the best performing weights. Use "
                                "the 'ModelCheckpoint' Callback to restore the best epoch weights."
                            )

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

    def _set_lr_scheduler_running_params(
        self, lr_scheduler: Optional[LRScheduler], **kwargs
    ):
        reducelronplateau_criterion = kwargs.get("reducelronplateau_criterion", None)
        self._set_reduce_on_plateau_criterion(lr_scheduler, reducelronplateau_criterion)
        if lr_scheduler is not None:
            self.cyclic_lr = "cycl" in lr_scheduler.__class__.__name__.lower()
        else:
            self.cyclic_lr = False

    # this needs type fixing to adjust for the fact that the main class can
    # take an 'object', a non-instastiated Class, so, should be something like:
    # callbacks: Optional[List[Union[object, Callback]]] in all places
    def _set_callbacks_and_metrics(self, callbacks: Any, metrics: Any):
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
        default_num_workers = (
            0
            if sys.platform == "darwin" and sys.version_info.minor > 7
            else os.cpu_count()
        )
        default_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device = kwargs.get("device", default_device)
        num_workers = kwargs.get("num_workers", default_num_workers)
        return device, num_workers

    def __repr__(self) -> str:
        list_of_params: List[str] = []
        list_of_params.append(f"fname={self.model.__class__.__name__}")
        list_of_params.append("objective={objective}")
        list_of_params.append(f"custom_loss_function={self.loss_fn.__class__.__name__}")
        list_of_params.append(f"optimizer={self.optimizer.__class__.__name__}")
        if self.lr_scheduler is not None:
            list_of_params.append(
                f"lr_scheduler={self.lr_scheduler.__class__.__name__}"
            )
        if self.callbacks is not None:
            callbacks = (
                "[" + ", ".join([c.__class__.__name__ for c in self.callbacks]) + "]"
            )
            list_of_params.append(f"callbacks={callbacks}")
        if self.verbose is not None:
            list_of_params.append("verbose={verbose}")
        if self.seed is not None:
            list_of_params.append("seed={seed}")
        if self.device is not None:
            list_of_params.append("device={device}")
        if self.num_workers is not None:
            list_of_params.append("num_workers={num_workers}")
        all_params = ", ".join(list_of_params)
        return f"BayesianTrainer({all_params.format(**self.__dict__)})"
