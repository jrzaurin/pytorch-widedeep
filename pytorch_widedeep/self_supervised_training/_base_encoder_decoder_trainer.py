import os
import sys
import warnings
from abc import ABC, abstractmethod

import numpy as np
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

from pytorch_widedeep.losses import EncoderDecoderLoss
from pytorch_widedeep.wdtypes import (
    Any,
    List,
    Optional,
    Optimizer,
    LRScheduler,
    ModelWithoutAttention,
    DecoderWithoutAttention,
)
from pytorch_widedeep.callbacks import (
    History,
    Callback,
    CallbackContainer,
    LRShedulerCallback,
)
from pytorch_widedeep.models.tabular.self_supervised import EncoderDecoderModel


class BaseEncoderDecoderTrainer(ABC):
    def __init__(
        self,
        encoder: ModelWithoutAttention,
        decoder: DecoderWithoutAttention,
        masked_prob: float,
        optimizer: Optional[Optimizer],
        lr_scheduler: Optional[LRScheduler],
        callbacks: Optional[List[Callback]],
        verbose: int,
        seed: int,
        **kwargs,
    ):
        self.device, self.num_workers = self._set_device_and_num_workers(**kwargs)

        self.early_stop = False
        self.verbose = verbose
        self.seed = seed

        self.ed_model = EncoderDecoderModel(
            encoder,
            decoder,
            masked_prob,
        )
        self.ed_model.to(self.device)

        self.loss_fn = EncoderDecoderLoss()
        self.optimizer = (
            optimizer
            if optimizer is not None
            else torch.optim.AdamW(self.ed_model.parameters())
        )
        self.lr_scheduler = lr_scheduler
        self._set_lr_scheduler_running_params(lr_scheduler, **kwargs)
        self._set_callbacks(callbacks)

    @abstractmethod
    def pretrain(
        self,
        X_tab: np.ndarray,
        X_val: Optional[np.ndarray],
        val_split: Optional[float],
        validation_freq: int,
        n_epochs: int,
        batch_size: int,
    ):
        raise NotImplementedError("Trainer.pretrain method not implemented")

    @abstractmethod
    def save(
        self,
        path: str,
        save_state_dict: bool,
        model_filename: str,
    ):
        raise NotImplementedError("Trainer.save method not implemented")

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
            self.reducelronplateau_criterion = "loss"
        else:
            self.reducelronplateau_criterion = reducelronplateau_criterion

    def _set_lr_scheduler_running_params(self, lr_scheduler, **kwargs):
        reducelronplateau_criterion = kwargs.get("reducelronplateau_criterion", None)
        self._set_reduce_on_plateau_criterion(lr_scheduler, reducelronplateau_criterion)
        if lr_scheduler is not None:
            self.cyclic_lr = "cycl" in lr_scheduler.__class__.__name__.lower()
        else:
            self.cyclic_lr = False

    def _set_callbacks(self, callbacks: Any):
        self.callbacks: List = [History(), LRShedulerCallback()]
        if callbacks is not None:
            for callback in callbacks:
                if isinstance(callback, type):
                    callback = callback()
                self.callbacks.append(callback)
        self.callback_container = CallbackContainer(self.callbacks)
        self.callback_container.set_model(self.ed_model)
        self.callback_container.set_trainer(self)

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
                        self.ed_model.load_state_dict(callback.best_state_dict)
                    else:
                        if self.verbose:
                            print(
                                "Model weights after training corresponds to the those of the "
                                "final epoch which might not be the best performing weights. Use "
                                "the 'ModelCheckpoint' Callback to restore the best epoch weights."
                            )

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
        list_of_params.append(f"encoder={self.ed_model.encoder.__class__.__name__}")
        list_of_params.append(f"decoder={self.ed_model.decoder.__class__.__name__}")
        list_of_params.append(f"masked_prob={self.ed_model.masker.p}")
        if self.optimizer is not None:
            list_of_params.append(f"optimizer={self.optimizer.__class__.__name__}")
        if self.lr_scheduler is not None:
            list_of_params.append(
                f"lr_scheduler={self.lr_scheduler.__class__.__name__}"
            )
        if self.callbacks is not None:
            callbacks = [c.__class__.__name__ for c in self.callbacks]
            list_of_params.append(f"callbacks={callbacks}")
        list_of_params.append("verbose={verbose}")
        list_of_params.append("seed={seed}")
        all_params = ", ".join(list_of_params)
        return f"EncoderDecoderTrainer({all_params.format(**self.__dict__)})"
