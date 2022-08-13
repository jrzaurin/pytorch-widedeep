import os
import sys
from abc import ABC, abstractmethod

import numpy as np
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

from pytorch_widedeep.losses import EncoderDecoderLoss
from pytorch_widedeep.wdtypes import (
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

    def _set_callbacks(self, callbacks):
        self.callbacks: List = [History(), LRShedulerCallback()]
        if callbacks is not None:
            for callback in callbacks:
                if isinstance(callback, type):
                    callback = callback()
                self.callbacks.append(callback)
        self.callback_container = CallbackContainer(self.callbacks)
        self.callback_container.set_model(self.ed_model)
        self.callback_container.set_trainer(self)

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
