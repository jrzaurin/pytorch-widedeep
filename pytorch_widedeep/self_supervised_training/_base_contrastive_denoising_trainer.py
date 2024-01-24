import os
import sys
import warnings
from abc import ABC, abstractmethod

import numpy as np
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

from pytorch_widedeep.losses import InfoNCELoss, DenoisingLoss
from pytorch_widedeep.wdtypes import (
    Any,
    List,
    Tuple,
    Tensor,
    Literal,
    Optional,
    Optimizer,
    LRScheduler,
    ModelWithAttention,
)
from pytorch_widedeep.callbacks import (
    History,
    Callback,
    CallbackContainer,
    LRShedulerCallback,
)
from pytorch_widedeep.models.tabular.self_supervised import (
    ContrastiveDenoisingModel,
)
from pytorch_widedeep.preprocessing.tab_preprocessor import TabPreprocessor


class BaseContrastiveDenoisingTrainer(ABC):
    def __init__(
        self,
        model: ModelWithAttention,
        preprocessor: TabPreprocessor,
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
        self._check_projection_head_dims(
            model, projection_head1_dims, projection_head2_dims
        )
        self._check_model_is_supported(model)
        self.device, self.num_workers = self._set_device_and_num_workers(**kwargs)

        self.early_stop = False
        self.verbose = verbose
        self.seed = seed

        self.cd_model = ContrastiveDenoisingModel(
            model,
            preprocessor,
            loss_type,
            projection_head1_dims,
            projection_head2_dims,
            projection_heads_activation,
            cat_mlp_type,
            cont_mlp_type,
            denoise_mlps_activation,
        )
        self.cd_model.to(self.device)

        self.loss_type = loss_type
        self._set_loss_fn(**kwargs)
        self.optimizer = (
            optimizer
            if optimizer is not None
            else torch.optim.AdamW(self.cd_model.parameters())
        )
        self.lr_scheduler = lr_scheduler
        self._set_lr_scheduler_running_params(lr_scheduler, **kwargs)
        self._set_callbacks(callbacks)

    @abstractmethod
    def pretrain(
        self,
        X_tab: np.ndarray,
        X_tab_val: Optional[np.ndarray],
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

    def _set_loss_fn(self, **kwargs):
        if self.loss_type in ["contrastive", "both"]:
            temperature = kwargs.get("temperature", 0.1)
            reduction = kwargs.get("reduction", "mean")
            self.contrastive_loss = InfoNCELoss(temperature, reduction)

        if self.loss_type in ["denoising", "both"]:
            lambda_cat = kwargs.get("lambda_cat", 1.0)
            lambda_cont = kwargs.get("lambda_cont", 1.0)
            reduction = kwargs.get("reduction", "mean")
            self.denoising_loss = DenoisingLoss(lambda_cat, lambda_cont, reduction)

    def _compute_loss(
        self,
        g_projs: Optional[Tuple[Tensor, Tensor]],
        x_cat_and_cat_: Optional[Tuple[Tensor, Tensor]],
        x_cont_and_cont_: Optional[Tuple[Tensor, Tensor]],
    ) -> Tensor:
        contrastive_loss = (
            self.contrastive_loss(g_projs)
            if self.loss_type in ["contrastive", "both"]
            else torch.tensor(0.0)
        )
        denoising_loss = (
            self.denoising_loss(x_cat_and_cat_, x_cont_and_cont_)
            if self.loss_type in ["denoising", "both"]
            else torch.tensor(0.0)
        )

        return contrastive_loss + denoising_loss

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
        self.callback_container.set_model(self.cd_model)
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
                        self.cd_model.load_state_dict(callback.best_state_dict)
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

    @staticmethod
    def _check_model_is_supported(model: ModelWithAttention):
        if model.__class__.__name__ == "TabPerceiver":
            raise ValueError(
                "Self-Supervised pretraining is not supported for the 'TabPerceiver'"
            )
        if model.__class__.__name__ == "TabTransformer" and not model.embed_continuous:
            raise ValueError(
                "Self-Supervised pretraining is only supported if both categorical and "
                "continuum columns are embedded. Please set 'embed_continuous = True'"
            )

    @staticmethod
    def _check_projection_head_dims(
        model: ModelWithAttention,
        projection_head1_dims: Optional[List[int]],
        projection_head2_dims: Optional[List[int]],
    ):
        error_msg = (
            "The first dimension of the projection heads must be the same as "
            f"the embeddings dimension or input dimension of the model: {model.input_dim}. "
        )

        if (
            projection_head1_dims is not None
            and model.input_dim != projection_head1_dims[0]
        ):
            raise ValueError(error_msg)

        if (
            projection_head2_dims is not None
            and model.input_dim != projection_head2_dims[0]
        ):
            raise ValueError(error_msg)

    def __repr__(self) -> str:
        list_of_params: List[str] = []
        list_of_params.append(f"model={self.cd_model.__class__.__name__}")
        if self.optimizer is not None:
            list_of_params.append(f"optimizer={self.optimizer.__class__.__name__}")
        if self.lr_scheduler is not None:
            list_of_params.append(
                f"lr_scheduler={self.lr_scheduler.__class__.__name__}"
            )
        if self.callbacks is not None:
            callbacks = [c.__class__.__name__ for c in self.callbacks]
            list_of_params.append(f"callbacks={callbacks}")
        list_of_params.append("loss_type={loss_type}")
        if self.cd_model.projection_head1_dims is not None:
            list_of_params.append(
                f"projection_head1_dims={self.cd_model.projection_head1_dims}"
            )
        if self.cd_model.projection_head2_dims is not None:
            list_of_params.append(
                f"projection_head2_dims={self.cd_model.projection_head2_dims}"
            )
        list_of_params.append(
            f"projection_heads_activation={self.cd_model.projection_heads_activation}"
        )
        list_of_params.append(f"cat_mlp_type={self.cd_model.cat_mlp_type}")
        list_of_params.append(f"cont_mlp_type={self.cd_model.cont_mlp_type}")
        list_of_params.append(
            f"denoise_mlps_activation={self.cd_model.denoise_mlps_activation}"
        )
        list_of_params.append("verbose={verbose}")
        list_of_params.append("seed={seed}")
        all_params = ", ".join(list_of_params)
        return f"ContrastiveDenoisingTrainer({all_params.format(**self.__dict__)})"
