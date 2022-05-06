import os
import sys
from abc import ABC, abstractmethod

import torch

from pytorch_widedeep.losses import InfoNCELoss, DenoisingLoss
from pytorch_widedeep.wdtypes import *  # noqa: F403
from pytorch_widedeep.callbacks import (
    History,
    Callback,
    CallbackContainer,
    LRShedulerCallback,
)
from pytorch_widedeep.self_supervised_training.self_supervised_model import (
    SelfSupervisedModel,
)


class BaseSelfSupervisedTrainer(ABC):
    def __init__(
        self,
        model,
        optimizer: Optional[Optimizer],
        lr_scheduler: Optional[LRScheduler],
        callbacks: Optional[List[Callback]],
        loss_type: Literal["contrastive", "denoising", "both"],
        projection_head1_dims: Optional[List[int]],
        projection_head2_dims: Optional[List[int]],
        projection_heads_activation: str,
        cat_mlp_type: Literal["single", "multiple"],
        cont_mlp_type: Literal["single", "multiple"],
        denoise_mlps_activation: str,
        verbose: int,
        seed: int,
        **kwargs,
    ):

        self.ss_model = SelfSupervisedModel(
            model,
            loss_type,
            projection_head1_dims,
            projection_head2_dims,
            projection_heads_activation,
            cat_mlp_type,
            cont_mlp_type,
            denoise_mlps_activation,
        )

        self.device, self.num_workers = self._set_device_and_num_workers(**kwargs)

        self.early_stop = False
        self.ss_model.to(self.device)

        self.loss_type = loss_type
        self._set_loss_fn(**kwargs)

        self.verbose = verbose
        self.seed = seed

        self.optimizer = (
            optimizer
            if optimizer is not None
            else torch.optim.AdamW(self.ss_model.parameters())
        )
        self.lr_scheduler = self._set_lr_scheduler_running_params(
            lr_scheduler, **kwargs
        )
        self._set_callbacks(callbacks)

    @abstractmethod
    def pretrain(self):
        pass

    def _set_loss_fn(self, **kwargs):

        if self.loss_type in ["contrastive", "both"]:
            temperature: float = kwargs.get("temperature", 0.1)
            reductiom: str = kwargs.get("reductiom", "mean")
            self.contrastive_loss = InfoNCELoss(temperature, reductiom)

        if self.loss_type in ["denoising", "both"]:
            lambda_cat: float = kwargs.get("lambda_cat", 1.0)
            lambda_cont: float = kwargs.get("lambda_cont", 1.0)
            reductiom: str = kwargs.get("reductiom", "mean")
            self.denoising_loss = DenoisingLoss(lambda_cat, lambda_cont, reductiom)

    def _compute_loss(
        self,
        g_projs: Optional[Tuple[Tensor, Tensor]],
        x_cat_and_cat_: Optional[Tuple[Tensor, Tensor]],
        x_cont_and_cont_: Optional[Tuple[Tensor, Tensor]],
    ) -> Tensor:

        contrastive_loss = (
            self.contrastive_loss(g_projs) if g_projs is not None else torch.tensor(0.0)
        )
        denoising_loss = self.denoising_loss(x_cat_and_cat_, x_cont_and_cont_)

        return contrastive_loss + denoising_loss

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
        self.callback_container.set_model(self.ss_model)
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
                        self.ss_model.load_state_dict(callback.best_state_dict)
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
