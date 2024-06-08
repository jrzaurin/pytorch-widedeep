import json
from pathlib import Path

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
    Literal,
    Optional,
    WideDeep,
    Optimizer,
    Transforms,
    LRScheduler,
)
from pytorch_widedeep.callbacks import Callback
from pytorch_widedeep.initializers import Initializer
from pytorch_widedeep.training._finetune import FineTune
from pytorch_widedeep.utils.general_utils import alias
from pytorch_widedeep.training._wd_dataset import WideDeepDataset
from pytorch_widedeep.training._base_trainer import BaseTrainer
from pytorch_widedeep.training._trainer_utils import (
    save_epoch_logs,
    print_loss_and_metric,
)

# Observation 1: I am annoyed by sublime highlighting an override issue with
# the abstractmethods. There is no override issue. The Signature of
# the 'predict' method is compatible with the supertype. Buy for whatever
# issue sublime highlights this as an error (not vscode and is not returned
# as an error when running mypy). I am ignoring it

# There is a lot of code repetition between this class and the 'Trainer'
# class (and in consquence a lot of ignore methods for test coverage). Maybe
# in the future I decided to merge the two of them and offer the ability to
# laod from folder based on the input parameters. For now, I'll leave it like
# this, separated, since it is the easiest and most manageable(i.e. easier to
# debug) implementation


class TrainerFromFolder(BaseTrainer):
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
             string indicating the device. One of _'cpu'_ or _'gpu'_

         - **num_workers**: `int`<br/>
             number of workers to be used internally by the data loaders

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
            verbose=verbose,
            seed=seed,
            **kwargs,
        )

    @alias("finetune", ["warmup"])
    def fit(  # noqa: C901
        self,
        train_loader: DataLoader,
        eval_loader: Optional[DataLoader] = None,
        n_epochs: int = 1,
        validation_freq: int = 1,
        finetune: bool = False,
        **kwargs,
    ):
        finetune_args = self._extract_kwargs(kwargs)

        train_steps = len(train_loader)

        if finetune:
            self._finetune(train_loader, **finetune_args)
            if self.verbose:
                print(
                    "Fine-tuning (or warmup) of individual components completed. "
                    "Training the whole model for {} epochs".format(n_epochs)
                )

        self.callback_container.on_train_begin(
            {
                "batch_size": train_loader.batch_size,
                "train_steps": train_steps,
                "n_epochs": n_epochs,
            }
        )
        for epoch in range(n_epochs):
            epoch_logs: Dict[str, float] = {}
            self.callback_container.on_epoch_begin(epoch, logs=epoch_logs)

            self.train_running_loss = 0.0
            with trange(train_steps, disable=self.verbose != 1) as t:
                for batch_idx, (data, targett) in zip(t, train_loader):
                    t.set_description("epoch %i" % (epoch + 1))
                    train_score, train_loss = self._train_step(
                        data, targett, batch_idx, epoch
                    )
                    print_loss_and_metric(t, train_loss, train_score)
                    self.callback_container.on_batch_end(batch=batch_idx)
            epoch_logs = save_epoch_logs(epoch_logs, train_loss, train_score, "train")

            on_epoch_end_metric = None
            if eval_loader is not None and epoch % validation_freq == (
                validation_freq - 1
            ):
                eval_steps = len(eval_loader)
                self.callback_container.on_eval_begin()
                self.valid_running_loss = 0.0
                with trange(eval_steps, disable=self.verbose != 1) as v:
                    for i, (data, targett) in zip(v, eval_loader):
                        v.set_description("valid")
                        val_score, val_loss = self._eval_step(data, targett, i)
                        print_loss_and_metric(v, val_loss, val_score)
                epoch_logs = save_epoch_logs(epoch_logs, val_loss, val_score, "val")

                if self.reducelronplateau:  # pragma: no cover
                    if self.reducelronplateau_criterion == "loss":
                        on_epoch_end_metric = val_loss
                    else:
                        on_epoch_end_metric = val_score[
                            self.reducelronplateau_criterion
                        ]
            else:
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

    def predict_uncertainty(  # type: ignore[return]
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

    def save(
        self,
        path: str,
        save_state_dict: bool = False,
        model_filename: str = "wd_model.pt",
    ):  # pragma: no cover
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

    @alias("n_epochs", ["finetune_epochs", "warmup_epochs"])
    @alias("max_lr", ["finetune_max_lr", "warmup_max_lr"])
    def _finetune(
        self,
        loader: DataLoader,
        n_epochs: int = 5,
        max_lr: float = 0.01,
        routine: Literal["howard", "felbo"] = "howard",
        deeptabular_gradual: bool = False,
        deeptabular_layers: Optional[List[nn.Module]] = None,
        deeptabular_max_lr: float = 0.01,
        deeptext_gradual: bool = False,
        deeptext_layers: Optional[List[nn.Module]] = None,
        deeptext_max_lr: float = 0.01,
        deepimage_gradual: bool = False,
        deepimage_layers: Optional[List[nn.Module]] = None,
        deepimage_max_lr: float = 0.01,
    ):
        r"""
        Simple wrap-up to individually fine-tune model components
        """
        if self.model.deephead is not None:
            raise ValueError(
                "Currently warming up is only supported without a fully connected 'DeepHead'"
            )

        finetuner = FineTune(self.loss_fn, self.metric, self.method, self.verbose)  # type: ignore[arg-type]
        if self.model.wide:
            finetuner.finetune_all(self.model.wide, "wide", loader, n_epochs, max_lr)

        if self.model.deeptabular:
            if deeptabular_gradual:
                assert (
                    deeptabular_layers is not None
                ), "deeptabular_layers must be passed if deeptabular_gradual=True"
                finetuner.finetune_gradual(
                    self.model.deeptabular,
                    "deeptabular",
                    loader,
                    deeptabular_max_lr,
                    deeptabular_layers,
                    routine,
                )
            else:
                finetuner.finetune_all(
                    self.model.deeptabular, "deeptabular", loader, n_epochs, max_lr
                )

        if self.model.deeptext:
            if deeptext_gradual:
                assert (
                    deeptext_layers is not None
                ), "deeptext_layers must be passed if deeptabular_gradual=True"
                finetuner.finetune_gradual(
                    self.model.deeptext,
                    "deeptext",
                    loader,
                    deeptext_max_lr,
                    deeptext_layers,
                    routine,
                )
            else:
                finetuner.finetune_all(
                    self.model.deeptext, "deeptext", loader, n_epochs, max_lr
                )

        if self.model.deepimage:
            if deepimage_gradual:
                assert (
                    deepimage_layers is not None
                ), "deepimage_layers must be passed if deeptabular_gradual=True"
                finetuner.finetune_gradual(
                    self.model.deepimage,
                    "deepimage",
                    loader,
                    deepimage_max_lr,
                    deepimage_layers,
                    routine,
                )
            else:
                finetuner.finetune_all(
                    self.model.deepimage, "deepimage", loader, n_epochs, max_lr
                )

    def _train_step(
        self,
        data: Dict[str, Union[Tensor, List[Tensor]]],
        target: Tensor,
        batch_idx: int,
        epoch: int,
    ):
        self.model.train()
        X: Dict[str, Union[Tensor, List[Tensor]]] = {}
        for k, v in data.items():
            if isinstance(v, list):
                X[k] = [i.to(self.device) for i in v]
            else:
                X[k] = v.to(self.device)
        y = (
            target.view(-1, 1).float()
            if self.method not in ["multiclass", "qregression"]
            else target
        )
        y = y.to(self.device)

        self.optimizer.zero_grad()

        y_pred = self.model(X)

        if self.model.is_tabnet:  # pragma: no cover
            loss = self.loss_fn(y_pred[0], y) - self.lambda_sparse * y_pred[1]
            score = self._get_score(y_pred[0], y)
        else:
            loss = self.loss_fn(y_pred, y)
            score = self._get_score(y_pred, y)

        loss.backward()
        self.optimizer.step()

        self.train_running_loss += loss.item()
        avg_loss = self.train_running_loss / (batch_idx + 1)

        return score, avg_loss

    def _eval_step(self, data: Dict[str, Tensor], target: Tensor, batch_idx: int):
        self.model.eval()
        with torch.no_grad():
            X: Dict[str, Union[Tensor, List[Tensor]]] = {}
            for k, v in data.items():
                if isinstance(v, list):
                    X[k] = [i.to(self.device) for i in v]
                else:
                    X[k] = v.to(self.device)
            y = (
                target.view(-1, 1).float()
                if self.method not in ["multiclass", "qregression"]
                else target
            )
            y = y.to(self.device)

            y_pred = self.model(X)
            if self.model.is_tabnet:  # pragma: no cover
                loss = self.loss_fn(y_pred[0], y) - self.lambda_sparse * y_pred[1]
                score = self._get_score(y_pred[0], y)
            else:
                score = self._get_score(y_pred, y)
                loss = self.loss_fn(y_pred, y)

            self.valid_running_loss += loss.item()
            avg_loss = self.valid_running_loss / (batch_idx + 1)

        return score, avg_loss

    def _get_score(self, y_pred, y):  # pragma: no cover
        if self.metric is not None:
            if self.method == "regression":
                score = self.metric(y_pred, y)
            if self.method == "binary":
                score = self.metric(torch.sigmoid(y_pred), y)
            if self.method == "qregression":
                score = self.metric(y_pred, y)
            if self.method == "multiclass":
                score = self.metric(F.softmax(y_pred, dim=1), y)
            return score
        else:
            return None

    def _predict(  # noqa: C901
        self,
        X_wide: Optional[np.ndarray] = None,
        X_tab: Optional[np.ndarray] = None,
        X_text: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
        X_img: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
        X_test: Optional[Dict[str, Union[np.ndarray, List[np.ndarray]]]] = None,
        test_loader: Optional[DataLoader] = None,
        batch_size: Optional[int] = None,
        uncertainty_granularity=1000,
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
                                    X[k] = [i.to(self.device) for i in v]
                                else:
                                    X[k] = v.to(self.device)
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

    @staticmethod
    def _predict_ziln(preds: Tensor) -> Tensor:  # pragma: no cover
        """Calculates predicted mean of zero inflated lognormal logits.

        Adjusted implementaion of `code
        <https://github.com/google/lifetime_value/blob/master/lifetime_value/zero_inflated_lognormal.py>`

        Arguments:
            preds: [batch_size, 3] tensor of logits.
        Returns:
            ziln_preds: [batch_size, 1] tensor of predicted mean.
        """
        positive_probs = torch.sigmoid(preds[..., :1])
        loc = preds[..., 1:2]
        scale = F.softplus(preds[..., 2:])
        ziln_preds = positive_probs * torch.exp(loc + 0.5 * torch.square(scale))
        return ziln_preds

    @staticmethod
    def _extract_kwargs(kwargs):
        finetune_params = [
            "n_epochs",
            "finetune_epochs",
            "warmup_epochs",
            "max_lr",
            "finetune_max_lr",
            "warmup_max_lr",
            "routine",
            "deeptabular_gradual",
            "deeptabular_layers",
            "deeptabular_max_lr",
            "deeptext_gradual",
            "deeptext_layers",
            "deeptext_max_lr",
            "deepimage_gradual",
            "deepimage_layers",
            "deepimage_max_lr",
        ]

        finetune_args = {}
        for k, v in kwargs.items():
            if k in finetune_params:
                finetune_args[k] = v

        return finetune_args
