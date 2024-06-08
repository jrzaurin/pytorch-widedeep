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
    Dict,
    List,
    Union,
    Module,
    Optional,
    WideDeep,
    Optimizer,
    Transforms,
    LRScheduler,
)
from pytorch_widedeep.callbacks import (
    History,
    Callback,
    MetricCallback,
    CallbackContainer,
    LRShedulerCallback,
)
from pytorch_widedeep.initializers import Initializer, MultipleInitializer
from pytorch_widedeep.training._trainer_utils import alias_to_loss
from pytorch_widedeep.training._multiple_optimizer import MultipleOptimizer
from pytorch_widedeep.training._multiple_transforms import MultipleTransforms
from pytorch_widedeep.training._loss_and_obj_aliases import _ObjectiveToMethod
from pytorch_widedeep.training._multiple_lr_scheduler import (
    MultipleLRScheduler,
)

# I would like to be more specific with the abstract methods in 'BaseTrainer'
# so they are more informative. However I also want to use this class to be
# inherited by the 'TrainerFromFolder' and the signatures of the methods are
# different.


class BaseTrainer(ABC):
    def __init__(
        self,
        model: WideDeep,
        objective: str,
        custom_loss_function: Optional[Module],
        optimizers: Optional[
            Union[Optimizer, Dict[str, Union[Optimizer, List[Optimizer]]]]
        ],
        lr_schedulers: Optional[
            Union[LRScheduler, Dict[str, Union[LRScheduler, List[LRScheduler]]]]
        ],
        initializers: Optional[
            Union[Initializer, Dict[str, Union[Initializer, List[Initializer]]]]
        ],
        transforms: Optional[List[Transforms]],
        callbacks: Optional[List[Callback]],
        metrics: Optional[Union[List[Metric], List[TorchMetric]]],
        verbose: int,
        seed: int,
        **kwargs,
    ):
        self._check_inputs(
            model, objective, optimizers, lr_schedulers, custom_loss_function
        )
        self.device, self.num_workers = self._set_device_and_num_workers(**kwargs)

        self.early_stop = False
        self.verbose = verbose
        self.seed = seed

        self.model = model
        if self.model.is_tabnet:
            self.lambda_sparse = kwargs.get("lambda_sparse", 1e-3)
        self.model.to(self.device)
        self.model.wd_device = self.device

        self.objective = objective
        self.method: str = _ObjectiveToMethod.get(objective)  # type: ignore

        if initializers is not None:
            self._initialize(initializers)
        self.loss_fn = self._set_loss_fn(objective, custom_loss_function, **kwargs)
        self.optimizer = self._set_optimizer(optimizers)
        self.lr_scheduler = self._set_lr_scheduler(lr_schedulers, **kwargs)
        self.transforms = self._set_transforms(transforms)
        self._set_callbacks_and_metrics(callbacks, metrics)

    @abstractmethod
    def fit(self, **kwargs):
        pass

    @abstractmethod
    def predict(self, *args) -> np.ndarray:
        pass

    @abstractmethod
    def predict_proba(self, *args) -> np.ndarray:
        pass

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

    def _initialize(self, initializers):
        if isinstance(initializers, Dict):
            self.initializer = MultipleInitializer(
                initializers, verbose=self.verbose > 0
            )
            self.initializer.apply(self.model)
        elif isinstance(initializers, type):
            self.initializer = initializers()
            self.initializer(self.model)
        elif isinstance(initializers, Initializer):
            self.initializer = initializers
            self.initializer(self.model)
        else:
            # aesthetics
            pass

    def _set_loss_fn(self, objective, custom_loss_function, **kwargs):
        class_weight = (
            torch.tensor(kwargs["class_weight"]).to(self.device)
            if "class_weight" in kwargs
            else None
        )

        if custom_loss_function is not None:
            return custom_loss_function
        elif (
            self.method not in ["regression", "qregression"]
            and "focal_loss" not in objective
        ):
            return alias_to_loss(objective, weight=class_weight)
        elif "focal_loss" in objective:
            alpha = kwargs.get("alpha", 0.25)
            gamma = kwargs.get("gamma", 2.0)
            return alias_to_loss(objective, alpha=alpha, gamma=gamma)
        else:
            return alias_to_loss(objective)

    def _set_optimizer(
        self,
        optimizers: Optional[
            Union[Optimizer, Dict[str, Union[Optimizer, List[Optimizer]]]]
        ],
    ):
        if optimizers is not None:
            if isinstance(optimizers, Optimizer):
                optimizer: Union[Optimizer, MultipleOptimizer] = optimizers
            elif isinstance(optimizers, Dict):
                opt_names = list(optimizers.keys())
                mod_names = [n for n, c in self.model.named_children()]
                # if with_fds - the prediction layer is part of the model and
                # should be optimized with the rest of deeptabular
                # component/model
                if self.model.with_fds:
                    if "enf_pos" in mod_names:
                        mod_names.remove("enf_pos")
                    mod_names.remove("fds_layer")
                    # The Tabular optimizer is always going to be just one
                    assert isinstance(optimizers["deeptabular"], Optimizer)
                    optimizers["deeptabular"].add_param_group(
                        {"params": self.model.fds_layer.pred_layer.parameters()}
                    )
                for mn in mod_names:
                    assert mn in opt_names, "No optimizer found for {}".format(mn)
                optimizer = MultipleOptimizer(optimizers)
        else:
            optimizer = torch.optim.Adam(self.model.parameters())  # type: ignore
        return optimizer

    def _set_lr_scheduler(
        self,
        lr_schedulers: Optional[
            Union[LRScheduler, Dict[str, Union[LRScheduler, List[LRScheduler]]]]
        ] = None,
        **kwargs,
    ) -> Optional[Union[LRScheduler, MultipleLRScheduler]]:
        # ReduceLROnPlateau is special
        reducelronplateau_criterion = kwargs.get("reducelronplateau_criterion", None)

        self._set_reduce_on_plateau_criterion(
            lr_schedulers, reducelronplateau_criterion
        )

        if lr_schedulers is not None:
            if isinstance(lr_schedulers, LRScheduler) or isinstance(
                lr_schedulers, ReduceLROnPlateau
            ):
                lr_scheduler: Optional[Union[LRScheduler, MultipleLRScheduler]] = (
                    lr_schedulers
                )
                cyclic_lr = "cycl" in lr_scheduler.__class__.__name__.lower()
            else:
                lr_scheduler = MultipleLRScheduler(lr_schedulers)
                scheduler_names: List[str] = []
                for _, sc in lr_scheduler._schedulers.items():
                    if isinstance(sc, list):
                        scheduler_names += [s.__class__.__name__.lower() for s in sc]
                    else:
                        scheduler_names.append(sc.__class__.__name__.lower())
                cyclic_lr = any(["cycl" in sn for sn in scheduler_names])
        else:
            lr_scheduler, cyclic_lr = None, False

        self.cyclic_lr = cyclic_lr

        return lr_scheduler

    def _set_reduce_on_plateau_criterion(
        self, lr_schedulers, reducelronplateau_criterion
    ):
        self.reducelronplateau = False

        if isinstance(lr_schedulers, Dict):
            for _, scheduler in lr_schedulers.items():
                if isinstance(scheduler, ReduceLROnPlateau):
                    self.reducelronplateau = True
        elif isinstance(lr_schedulers, ReduceLROnPlateau):
            self.reducelronplateau = True

        if self.reducelronplateau and not reducelronplateau_criterion:
            UserWarning(
                "The learning rate scheduler of at least one of the model components is of type "
                "ReduceLROnPlateau. The step method in this scheduler requires a 'metrics' param "
                "that can be either the validation loss or the validation metric. Please, when "
                "instantiating the Trainer, specify which quantity will be tracked using "
                "reducelronplateau_criterion = 'loss' (default) or reducelronplateau_criterion = 'metric'"
            )
            self.reducelronplateau_criterion = "loss"
        else:
            self.reducelronplateau_criterion = reducelronplateau_criterion

    @staticmethod
    def _set_transforms(transforms):
        if transforms is not None:
            return MultipleTransforms(transforms)()
        else:
            return None

    # this needs type fixing to adjust for the fact that the main class can
    # take an 'object', a non-instastiated Class, so, should be something like:
    # callbacks: Optional[List[Union[object, Callback]]] in all places
    def _set_callbacks_and_metrics(
        self,
        callbacks: Any,
        metrics: Any,
    ):
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
    def _check_inputs(
        model,
        objective,
        optimizers,
        lr_schedulers,
        custom_loss_function,
    ):
        if model.with_fds and _ObjectiveToMethod.get(objective) != "regression":
            raise ValueError(
                "Feature Distribution Smooting can be used only for regression"
            )

        if objective == "multitarget":
            assert custom_loss_function is not None, (
                "When 'objective' is 'multitarget', 'custom_loss_function' must be "
                "provided. This library provides three losses for multitarget problems: "
                "'MultiTargetRegressionLoss', 'MultiTargetClassificationLoss' and "
                "'MutilTargetRegressionAndClassificationLoss'. Any other loss function "
                "must be provided by the user. Please, read the documentation for more details"
            )

        if _ObjectiveToMethod.get(objective) == "multiclass" and model.pred_dim == 1:
            raise ValueError(
                "This is a multiclass classification problem but the size of the output layer"
                " is set to 1. Please, set the 'pred_dim' param equal to the number of classes "
                " when instantiating the 'WideDeep' class"
            )

        if isinstance(optimizers, Dict):
            if lr_schedulers is not None and not isinstance(lr_schedulers, Dict):
                raise ValueError(
                    "''optimizers' and 'lr_schedulers' must have consistent type: "
                    "(Optimizer and LRScheduler) or (Dict[str, Optimizer] and Dict[str, LRScheduler]) "
                    "Please, read the documentation or see the examples for more details"
                )

        if custom_loss_function is not None and objective not in [
            "binary",
            "multiclass",
            "regression",
            "multitarget",
        ]:
            raise ValueError(
                "If 'custom_loss_function' is not None, 'objective' must be 'binary' "
                "'multiclass', 'regression' or 'multitarget' consistent with the loss function"
            )

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
        default_device = "cuda" if torch.cuda.is_available() else "cpu"
        device = kwargs.get("device", default_device)
        num_workers = kwargs.get("num_workers", default_num_workers)
        return device, num_workers

    def __repr__(self) -> str:  # noqa: C901
        list_of_params: List[str] = []
        list_of_params.append(f"model={self.model.__class__.__name__}")
        list_of_params.append("objective={objective}")
        list_of_params.append(f"loss_function={self.loss_fn.__class__.__name__}")
        list_of_params.append(f"optimizers={self.optimizer.__class__.__name__}")
        list_of_params.append(f"lr_schedulers={self.lr_scheduler.__class__.__name__}")
        if self.callbacks is not None:
            callbacks_str = (
                "[" + ", ".join([c.__class__.__name__ for c in self.callbacks]) + "]"
            )
            list_of_params.append(f"callbacks={callbacks_str}")
        if self.verbose is not None:
            list_of_params.append("verbose={verbose}")
        if self.seed is not None:
            list_of_params.append("seed={seed}")
        if self.device is not None:
            list_of_params.append("device={device}")
        if self.num_workers is not None:
            list_of_params.append("num_workers={num_workers}")
        all_params = ", ".join(list_of_params)
        return f"Trainer({all_params.format(**self.__dict__)})"
