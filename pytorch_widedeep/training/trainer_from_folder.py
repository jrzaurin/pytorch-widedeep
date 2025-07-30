import numpy as np
import torch
import torch.nn.functional as F
from tqdm import trange
from torch import nn
from torchmetrics import Metric as TorchMetric
from torch.utils.data import DataLoader

from pytorch_widedeep.losses import ZILNLoss
from pytorch_widedeep.metrics import Metric
from pytorch_widedeep.wdtypes import (
    Dict,
    List,
    Union,
    Tensor,
    Optional,
    WideDeep,
    Optimizer,
    Transforms,
    LRScheduler,
)
from pytorch_widedeep.callbacks import Callback
from pytorch_widedeep.initializers import Initializer
from pytorch_widedeep.training.trainer import Trainer
from pytorch_widedeep.utils.general_utils import alias, to_device
from pytorch_widedeep.training._wd_dataset import WideDeepDataset


class TrainerFromFolder(Trainer):
    r"""Class to set the of attributes that will be used during the
     training process.

     For examples, please, see the examples folder in the repo.

     Parameters
     ----------
     model: `WideDeep`
         An object of class `WideDeep`
     objective: str
         Defines the objective, loss or cost function. <br/>

         Param aliases: `loss_function`, `loss_fn`, `loss`,
         `cost_function`, `cost_fn`, `cost`. <br/>

         Possible values are:

         - `binary`, aliases: `logistic`, `binary_logloss`, `binary_cross_entropy`

         - `binary_focal_loss`

         - `multiclass`, aliases: `multi_logloss`, `cross_entropy`, `categorical_cross_entropy`,

         - `multiclass_focal_loss`

         - `regression`, aliases: `mse`, `l2`, `mean_squared_error`

         - `mean_absolute_error`, aliases: `mae`, `l1`

         - `mean_squared_log_error`, aliases: `msle`

         - `root_mean_squared_error`, aliases:  `rmse`

         - `root_mean_squared_log_error`, aliases: `rmsle`

         - `zero_inflated_lognormal`, aliases: `ziln`

         - `quantile`

         - `tweedie`
     custom_loss_function: `nn.Module`. Optional, default = None
         It is possible to pass a custom loss function. See for example
         `pytorch_widedeep.losses.FocalLoss` for the required structure of the
         object or the Examples section in this documentation or in the repo.
         Note that if `custom_loss_function` is not `None`, `objective` must
         be _'binary'_, _'multiclass'_ or _'regression'_, consistent with the
         loss function
     optimizers: `Optimizer` or dict. Optional, default=None
         - An instance of Pytorch's `Optimizer` object
           (e.g. `torch.optim.Adam()`) or
         - a dictionary where there keys are the model components (i.e.
           _'wide'_, _'deeptabular'_, _'deeptext'_, _'deepimage'_
           and/or _'deephead'_)  and the values are the corresponding
           optimizers or list of optimizers if multiple models are used for
           the given data mode (e.g. two text columns/models for the deeptext
           component). If multiple optimizers are used the
           dictionary **MUST** contain an optimizer per model component.

         if no optimizers are passed it will default to `Adam` for all
         model components
     lr_schedulers: `LRScheduler` or dict. Optional, default=None
         - An instance of Pytorch's `LRScheduler` object (e.g
           `torch.optim.lr_scheduler.StepLR(opt, step_size=5)`) or
         - a dictionary where there keys are the model componenst (i.e. _'wide'_,
           _'deeptabular'_, _'deeptext'_, _'deepimage'_ and/or _'deephead'_) and the
           values are the corresponding learning rate schedulers or list of
             learning rate schedulers if multiple models are used for the given
             data mode (e.g. two text columns/models for the deeptext component).
     initializers: `Initializer` or dict. Optional, default=None
         - An instance of an `Initializer` object see `pytorch-widedeep.initializers` or
         - a dictionary where there keys are the model components (i.e. _'wide'_,
           _'deeptabular'_, _'deeptext'_, _'deepimage'_ and/or _'deephead'_)
           and the values are the corresponding initializers or list of
             initializers if multiple models are used for the given data mode (e.g.
             two text columns/models for the deeptext component).
    transforms: List. Optional, default=None
         List with `torchvision.transforms` to be applied to the image
         component of the model (i.e. `deepimage`) See
         [torchvision transforms](https://pytorch.org/docs/stable/torchvision/transforms.html).
     callbacks: List. Optional, default=None
         List with `Callback` objects. The three callbacks available in
         `pytorch-widedeep` are: `LRHistory`, `ModelCheckpoint` and
         `EarlyStopping`. The `History` and the `LRShedulerCallback` callbacks
         are used by default. This can also be a custom callback as long as
         the object of type `Callback`. See
         `pytorch_widedeep.callbacks.Callback` or the examples folder in the
         repo.
     metrics: List. Optional, default=None
         - List of objects of type `Metric`. Metrics available are:
           `Accuracy`, `Precision`, `Recall`, `FBetaScore`,
           `F1Score` and `R2Score`. This can also be a custom metric as long
           as it is an object of type `Metric`. See
           `pytorch_widedeep.metrics.Metric` or the examples folder in the
           repo
         - List of objects of type `torchmetrics.Metric`. This can be any
           metric from torchmetrics library
           [Examples](https://torchmetrics.readthedocs.io/en/latest/).
           This can also be a custom metric as long as
           it is an object of type `Metric`. See
           [the instructions](https://torchmetrics.readthedocs.io/en/latest/).
     verbose: int, default=1
         Verbosity level. If set to 0 nothing will be printed during training
     seed: int, default=1
         Random seed to be used internally for train/test split

     Other Parameters
     ----------------
     **kwargs: dict
         Other infrequently used arguments that can also be passed as kwargs are:

        - **device**: `str`<br/>
            string indicating the device. One of _'cpu'_, _'gpu'_ or 'mps' if
            run on a Mac with Apple silicon or AMD GPU(s)

         - **num_workers**: `int`<br/>
             number of workers to be used internally by the data loaders

        - **use_multi_gpu**: `bool`<br/>
            If True, the model will be trained on multiple GPUs. This is
            only supported for the `deeptabular` component.

            NOTE: this is an experimental feature and might not work as expected
            in some cases. While for the `Trainer` class, it has been extensively
            tested, for the `TrainerFromFolder` class, it has not been tested
            that thoroughly (in principle the `TrainerFromFolder` inherits from
            the `Trainer` class, so it should work).

        - **lambda_sparse**: `float`<br/>
             lambda sparse parameter in case the `deeptabular` component is `TabNet`

         - **class_weight**: `List[float]`<br/>
             This is the `weight` or `pos_weight` parameter in
             `CrossEntropyLoss` and `BCEWithLogitsLoss`, depending on whether
         - **reducelronplateau_criterion**: `str`
             This sets the criterion that will be used by the lr scheduler to
             take a step: One of _'loss'_ or _'metric'_. The ReduceLROnPlateau
             learning rate is a bit particular.

     Attributes
     ----------
     cyclic_lr: bool
         Attribute that indicates if any of the lr_schedulers is cyclic_lr
         (i.e. `CyclicLR` or
         `OneCycleLR`). See [Pytorch schedulers](https://pytorch.org/docs/stable/optim.html).

    """

    @alias(  # noqa: C901
        "objective",
        ["loss_function", "loss_fn", "loss", "cost_function", "cost_fn", "cost"],
    )
    @alias("metrics", ["train_metrics"])
    def __init__(
        self,
        model: WideDeep,
        objective: str,
        custom_loss_function: Optional[nn.Module] = None,
        optimizers: Optional[
            Union[Optimizer, Dict[str, Union[Optimizer, List[Optimizer]]]]
        ] = None,
        lr_schedulers: Optional[
            Union[LRScheduler, Dict[str, Union[LRScheduler, List[LRScheduler]]]]
        ] = None,
        initializers: Optional[
            Union[Initializer, Dict[str, Union[Initializer, List[Initializer]]]]
        ] = None,
        transforms: Optional[List[Transforms]] = None,
        callbacks: Optional[List[Callback]] = None,
        metrics: Optional[Union[List[Metric], List[TorchMetric]]] = None,
        eval_metrics: Optional[Union[List[Metric], List[TorchMetric]]] = None,
        verbose: int = 1,
        seed: int = 1,
        **kwargs,
    ):
        super().__init__(
            model=model,
            objective=objective,
            custom_loss_function=custom_loss_function,
            optimizers=optimizers,
            lr_schedulers=lr_schedulers,
            initializers=initializers,
            transforms=transforms,
            callbacks=callbacks,
            metrics=metrics,
            eval_metrics=eval_metrics,
            verbose=verbose,
            seed=seed,
            **kwargs,
        )

        if self.method == "multitarget":
            raise NotImplementedError(
                "Training from folder is not supported for multitarget models"
            )

    @alias("finetune", ["warmup"])
    def fit(  # type: ignore[override]  # noqa: C901
        self,
        train_loader: DataLoader,
        eval_loader: Optional[DataLoader] = None,
        n_epochs: int = 1,
        validation_freq: int = 1,
        finetune: bool = False,
        stop_after_finetuning: bool = False,
        **kwargs,
    ):

        # There will never be dataloader_args when using 'TrainingFromFolder'
        # as the loaders are passed as arguments
        _, finetune_args = self._extract_kwargs(kwargs)

        if finetune:
            self.with_finetuning: bool = True
            self._do_finetune(train_loader, **finetune_args)
            if self.verbose and not stop_after_finetuning:
                print(
                    "Fine-tuning (or warmup) of individual components completed. "
                    "Training the whole model for {} epochs".format(n_epochs)
                )
        else:
            self.with_finetuning = False

        if stop_after_finetuning:
            if self.verbose:
                print("Stopping after finetuning")
            return

        self.callback_container.on_train_begin(
            {
                "batch_size": train_loader.batch_size,
                "train_steps": len(train_loader),
                "n_epochs": n_epochs,
            }
        )
        for epoch in range(n_epochs):
            epoch_logs = self._train_epoch(train_loader, epoch)
            if eval_loader is not None and epoch % validation_freq == (
                validation_freq - 1
            ):
                epoch_logs, on_epoch_end_metric = self._eval_epoch(
                    eval_loader, epoch_logs
                )
            else:
                on_epoch_end_metric = None
                if self.reducelronplateau:
                    raise NotImplementedError(
                        "ReduceLROnPlateau scheduler can be used only with validation data."
                    )
            self.callback_container.on_epoch_end(epoch, epoch_logs, on_epoch_end_metric)
            if self.early_stop:
                # self.callback_container.on_train_end(epoch_logs)
                break

        self.callback_container.on_train_end(epoch_logs)
        self._restore_best_weights()
        self.model.train()

    def predict(  # type: ignore[override, return]
        self,
        X_wide: Optional[np.ndarray] = None,
        X_tab: Optional[np.ndarray] = None,
        X_text: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
        X_img: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
        X_test: Optional[Dict[str, Union[np.ndarray, List[np.ndarray]]]] = None,
        test_loader: Optional[DataLoader] = None,
        batch_size: Optional[int] = None,
    ) -> np.ndarray:
        preds_l = self._predict(
            X_wide, X_tab, X_text, X_img, X_test, test_loader, batch_size
        )
        if self.method == "regression":
            return np.vstack(preds_l).squeeze(1)
        if self.method == "binary":  # pragma: no cover
            preds = np.vstack(preds_l).squeeze(1)
            return (preds > 0.5).astype("int")
        if self.method == "qregression":  # pragma: no cover
            return np.vstack(preds_l)
        if self.method == "multiclass":  # pragma: no cover
            preds = np.vstack(preds_l)
            return np.argmax(preds, 1)

    def predict_uncertainty(  # type: ignore[override, return]
        self,
        X_wide: Optional[np.ndarray] = None,
        X_tab: Optional[np.ndarray] = None,
        X_text: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
        X_img: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
        X_test: Optional[Dict[str, Union[np.ndarray, List[np.ndarray]]]] = None,
        batch_size: Optional[int] = None,
        test_loader: Optional[DataLoader] = None,
        uncertainty_granularity=1000,
    ) -> np.ndarray:  # pragma: no cover
        preds_l = self._predict(
            X_wide,
            X_tab,
            X_text,
            X_img,
            X_test,
            test_loader,
            batch_size,
            uncertainty_granularity,
            uncertainty=True,
        )
        preds = np.vstack(preds_l)
        samples_num = int(preds.shape[0] / uncertainty_granularity)
        if self.method == "regression":
            preds = preds.squeeze(1)
            preds = preds.reshape((uncertainty_granularity, samples_num))
            return np.array(
                (
                    preds.max(axis=0),
                    preds.min(axis=0),
                    preds.mean(axis=0),
                    preds.std(axis=0),
                )
            ).T
        if self.method == "qregression":
            raise ValueError(
                "Currently predict_uncertainty is not supported for qregression method"
            )
        if self.method == "binary":
            preds = preds.squeeze(1)
            preds = preds.reshape((uncertainty_granularity, samples_num))
            preds = preds.mean(axis=0)
            probs = np.zeros([preds.shape[0], 3])
            probs[:, 0] = 1 - preds
            probs[:, 1] = preds
            return probs
        if self.method == "multiclass":
            preds = preds.reshape(uncertainty_granularity, samples_num, preds.shape[1])
            preds = preds.mean(axis=0)
            preds = np.hstack((preds, np.vstack(np.argmax(preds, 1))))
            return preds

    def predict_proba(  # type: ignore[override, return]
        self,
        X_wide: Optional[np.ndarray] = None,
        X_tab: Optional[np.ndarray] = None,
        X_text: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
        X_img: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
        X_test: Optional[Dict[str, Union[np.ndarray, List[np.ndarray]]]] = None,
        test_loader: Optional[DataLoader] = None,
        batch_size: Optional[int] = None,
    ) -> np.ndarray:  # pragma: no cover
        preds_l = self._predict(
            X_wide, X_tab, X_text, X_img, X_test, test_loader, batch_size
        )
        if self.method == "binary":
            preds = np.vstack(preds_l).squeeze(1)
            probs = np.zeros([preds.shape[0], 2])
            probs[:, 0] = 1 - preds
            probs[:, 1] = preds
            return probs
        if self.method == "multiclass":
            return np.vstack(preds_l)

    def _predict(  # type: ignore[override]  # noqa: C901
        self,
        X_wide: Optional[np.ndarray] = None,
        X_tab: Optional[np.ndarray] = None,
        X_text: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
        X_img: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
        X_test: Optional[Dict[str, Union[np.ndarray, List[np.ndarray]]]] = None,
        test_loader: Optional[DataLoader] = None,
        batch_size: Optional[int] = None,
        uncertainty_granularity: int = 1000,
        uncertainty: bool = False,
    ) -> List:
        r"""Private method to avoid code repetition in predict and
        predict_proba. For parameter information, please, see the .predict()
        method documentation
        """

        if test_loader is not None:
            test_steps = len(test_loader)
        else:
            if X_test is not None:
                test_set = WideDeepDataset(**X_test)  # type: ignore[arg-type]
            else:
                load_dict: Dict[str, Union[np.ndarray, List[np.ndarray]]] = {}
                if X_wide is not None:
                    load_dict = {"X_wide": X_wide}
                if X_tab is not None:
                    load_dict.update({"X_tab": X_tab})
                if X_text is not None:
                    load_dict.update({"X_text": X_text})
                if X_img is not None:
                    load_dict.update({"X_img": X_img})
                test_set = WideDeepDataset(**load_dict)  # type: ignore[arg-type]

            if not hasattr(self, "batch_size"):
                assert batch_size is not None, (
                    "'batch_size' has not be previosly set in this Trainer and must be"
                    " specified via the 'batch_size' argument in this predict call"
                )
                self.batch_size = batch_size

            test_loader = DataLoader(
                dataset=test_set,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=False,
            )
            test_steps = (len(test_loader.dataset) // test_loader.batch_size) + 1  # type: ignore[arg-type]

        self.model.eval()
        preds_l = []

        if uncertainty:  # pragma: no cover
            for m in self.model.modules():
                if m.__class__.__name__.startswith("Dropout"):
                    m.train()
            prediction_iters = uncertainty_granularity
        else:
            prediction_iters = 1

        with torch.no_grad():
            with trange(uncertainty_granularity, disable=uncertainty is False) as t:
                for _, _ in zip(t, range(prediction_iters)):
                    t.set_description("predict_UncertaintyIter")

                    with trange(
                        test_steps, disable=self.verbose != 1 or uncertainty is True
                    ) as tt:
                        for _, data in zip(tt, test_loader):
                            tt.set_description("predict")
                            X: Dict[str, Union[Tensor, List[Tensor]]] = {}
                            for k, v in data.items():
                                if isinstance(v, list):
                                    X[k] = [to_device(i, self.device) for i in v]
                                else:
                                    X[k] = to_device(v, self.device)
                            preds = (
                                self.model(X)
                                if not self.model.is_tabnet
                                else self.model(X)[0]
                            )
                            if self.method == "binary":
                                preds = torch.sigmoid(preds)
                            if self.method == "multiclass":
                                preds = F.softmax(preds, dim=1)
                            if self.method == "regression" and isinstance(
                                self.loss_fn, ZILNLoss
                            ):
                                preds = self._predict_ziln(preds)
                            preds = preds.cpu().data.numpy()
                            preds_l.append(preds)
        self.model.train()
        return preds_l
