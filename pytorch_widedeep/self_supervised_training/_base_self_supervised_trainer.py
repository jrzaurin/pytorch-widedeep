import os
import sys
from abc import ABC, abstractmethod

import torch
from pytorch_widedeep.wdtypes import *  # noqa: F403
from pytorch_widedeep.callbacks import (
    History,
    Callback,
    CallbackContainer,
    LRShedulerCallback,
)


class BaseSelfSupervisedTrainer(ABC):
    def __init__(
        self,
        model,
        optimizer: Optimizer,
        lr_scheduler: LRScheduler,
        callbacks: Optional[List[Callback]],
        verbose: int,
        seed: int,
        **kwargs,
    ):

        self.device, self.num_workers = self._set_device_and_num_workers(**kwargs)

        self.model = model
        self.early_stop = False

        self.verbose = verbose
        self.seed = seed

        self.optimizer = (
            optimizer
            if optimizer is not None
            else torch.optim.AdamW(self.model.parameters())
        )
        self.lr_scheduler = lr_scheduler
        self._set_lr_scheduler_running_params(lr_scheduler, **kwargs)
        self._set_callbacks(callbacks)
        self.model.to(self.device)

    @abstractmethod
    def fit(self):
        pass

    def _set_lr_scheduler_running_params(self, lr_scheduler, **kwargs):
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
