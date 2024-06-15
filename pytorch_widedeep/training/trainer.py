import json
import warnings
from inspect import signature
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
    Tuple,
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
from pytorch_widedeep.dataloaders import DataLoaderDefault
from pytorch_widedeep.initializers import Initializer
from pytorch_widedeep.training._finetune import FineTune
from pytorch_widedeep.utils.general_utils import alias
from pytorch_widedeep.training._wd_dataset import WideDeepDataset
from pytorch_widedeep.training._base_trainer import BaseTrainer
from pytorch_widedeep.training._trainer_utils import (
    save_epoch_logs,
    wd_train_val_split,
    print_loss_and_metric,
)
from pytorch_widedeep.training._feature_importance import (
    Explainer,
    FeatureImportance,
)


class Trainer(BaseTrainer):
    r"""Class to set the of attributes that will be used during the
    training process.

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

        - `multitarget`, aliases: `multi_target`

        **NOTE**: For `multitarget` a custom loss function must be passed
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
    feature_importance: dict
        dict where the keys are the column names and the values are the
        corresponding feature importances. This attribute will only exist
        if the `deeptabular` component is a Tabnet model.

    Examples
    --------
    >>> import torch
    >>> from torchvision.transforms import ToTensor
    >>>
    >>> # wide deep imports
    >>> from pytorch_widedeep.callbacks import EarlyStopping, LRHistory
    >>> from pytorch_widedeep.initializers import KaimingNormal, KaimingUniform, Normal, Uniform
    >>> from pytorch_widedeep.models import TabResnet, Vision, BasicRNN, Wide, WideDeep
    >>> from pytorch_widedeep import Trainer
    >>>
    >>> embed_input = [(u, i, j) for u, i, j in zip(["a", "b", "c"][:4], [4] * 3, [8] * 3)]
    >>> column_idx = {k: v for v, k in enumerate(["a", "b", "c"])}
    >>> wide = Wide(10, 1)
    >>>
    >>> # build the model
    >>> deeptabular = TabResnet(blocks_dims=[8, 4], column_idx=column_idx, cat_embed_input=embed_input)
    >>> deeptext = BasicRNN(vocab_size=10, embed_dim=4, padding_idx=0)
    >>> deepimage = Vision()
    >>> model = WideDeep(wide=wide, deeptabular=deeptabular, deeptext=deeptext, deepimage=deepimage)
    >>>
    >>> # set optimizers and schedulers
    >>> wide_opt = torch.optim.Adam(model.wide.parameters())
    >>> deep_opt = torch.optim.AdamW(model.deeptabular.parameters())
    >>> text_opt = torch.optim.Adam(model.deeptext.parameters())
    >>> img_opt = torch.optim.AdamW(model.deepimage.parameters())
    >>>
    >>> wide_sch = torch.optim.lr_scheduler.StepLR(wide_opt, step_size=5)
    >>> deep_sch = torch.optim.lr_scheduler.StepLR(deep_opt, step_size=3)
    >>> text_sch = torch.optim.lr_scheduler.StepLR(text_opt, step_size=5)
    >>> img_sch = torch.optim.lr_scheduler.StepLR(img_opt, step_size=3)
    >>>
    >>> optimizers = {"wide": wide_opt, "deeptabular": deep_opt, "deeptext": text_opt, "deepimage": img_opt}
    >>> schedulers = {"wide": wide_sch, "deeptabular": deep_sch, "deeptext": text_sch, "deepimage": img_sch}
    >>>
    >>> # set initializers and callbacks
    >>> initializers = {"wide": Uniform, "deeptabular": Normal, "deeptext": KaimingNormal, "deepimage": KaimingUniform}
    >>> transforms = [ToTensor]
    >>> callbacks = [LRHistory(n_epochs=4), EarlyStopping]
    >>>
    >>> # set the trainer
    >>> trainer = Trainer(model, objective="regression", initializers=initializers, optimizers=optimizers,
    ... lr_schedulers=schedulers, callbacks=callbacks, transforms=transforms)
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
        X_wide: Optional[np.ndarray] = None,
        X_tab: Optional[np.ndarray] = None,
        X_text: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
        X_img: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
        X_train: Optional[Dict[str, Union[np.ndarray, List[np.ndarray]]]] = None,
        X_val: Optional[Dict[str, Union[np.ndarray, List[np.ndarray]]]] = None,
        val_split: Optional[float] = None,
        target: Optional[np.ndarray] = None,
        n_epochs: int = 1,
        validation_freq: int = 1,
        batch_size: int = 32,
        custom_dataloader: Optional[DataLoader] = None,
        feature_importance_sample_size: Optional[int] = None,
        finetune: bool = False,
        with_lds: bool = False,
        **kwargs,
    ):
        r"""Fit method.

        The input datasets can be passed either directly via numpy arrays
        (`X_wide`, `X_tab`, `X_text` or `X_img`) or alternatively, in
        dictionaries (`X_train` or `X_val`).

        Parameters
        ----------
        X_wide: np.ndarray, Optional. default=None
            Input for the `wide` model component.
            See `pytorch_widedeep.preprocessing.WidePreprocessor`
        X_tab: np.ndarray, Optional. default=None
            Input for the `deeptabular` model component.
            See `pytorch_widedeep.preprocessing.TabPreprocessor`
        X_text: Union[np.ndarray, List[np.ndarray]], Optional. default=None
            Input for the `deeptext` model component.
            See `pytorch_widedeep.preprocessing.TextPreprocessor`.
            If multiple text columns/models are used, this should be a list of
            numpy arrays
        X_img: np.ndarray, Optional. default=None
            Input for the `deepimage` model component.
            See `pytorch_widedeep.preprocessing.ImagePreprocessor`.
            If multiple image columns/models are used, this should be a list of
            numpy arrays
        X_train: Dict, Optional. default=None
            The training dataset can also be passed in a dictionary. Keys are
            _'X_wide'_, _'X_tab'_, _'X_text'_, _'X_img'_ and _'target'_. Values
            are the corresponding matrices. Note that of multiple text or image
            columns/models are used, the corresponding values should be lists
            of numpy arrays
        X_val: Dict, Optional. default=None
            The validation dataset can also be passed in a dictionary. Keys
            are _'X_wide'_, _'X_tab'_, _'X_text'_, _'X_img'_ and _'target'_.
            Values are the corresponding matrices. Note that of multiple text
            or image columns/models are used, the corresponding values should
            be lists of numpy arrays
        val_split: float, Optional. default=None
            train/val split fraction
        target: np.ndarray, Optional. default=None
            target values
        n_epochs: int, default=1
            number of epochs
        validation_freq: int, default=1
            epochs validation frequency
        batch_size: int, default=32
            batch size
        custom_dataloader: `DataLoader`, Optional, default=None
            object of class `torch.utils.data.DataLoader`. Available
            predefined dataloaders are in `pytorch-widedeep.dataloaders`.If
            `None`, a standard torch `DataLoader` is used.
        finetune: bool, default=False
            fine-tune individual model components. This functionality can also
            be used to 'warm-up' (and hence the alias `warmup`) individual
            components before the joined training starts, and hence its
            alias. See the Examples folder in the repo for more details

            `pytorch_widedeep` implements 3 fine-tune routines.

            - fine-tune all trainable layers at once. This routine is
              inspired by the work of Howard & Sebastian Ruder 2018 in their
              [ULMfit paper](https://arxiv.org/abs/1801.06146). Using a
              Slanted Triangular learing (see
              [Leslie N. Smith paper](https://arxiv.org/pdf/1506.01186.pdf) ) ,
              the process is the following: *i*) the learning rate will
              gradually increase for 10% of the training steps from max_lr/10
              to max_lr. *ii*) It will then gradually decrease to max_lr/10
              for the remaining 90% of the steps. The optimizer used in the
              process is `Adam`.

            and two gradual fine-tune routines, where only certain layers are
            trained at a time.

            - The so called `Felbo` gradual fine-tune rourine, based on the the
              Felbo et al., 2017 [DeepEmoji paper](https://arxiv.org/abs/1708.00524).
            - The `Howard` routine based on the work of Howard & Sebastian Ruder 2018 in their
              [ULMfit paper](https://arxiv.org/abs/1801.06146>).

            For details on how these routines work, please see the Examples
            section in this documentation and the Examples folder in the repo. <br/>
            Param Alias: `warmup`
        with_lds: bool, default=False
            Boolean indicating if Label Distribution Smoothing will be used. <br/>
            information_source: **NOTE**: We consider this feature absolutely
            experimental and we recommend the user to not use it unless the
            corresponding [publication](https://arxiv.org/abs/2102.09554) is
            well understood

        Other Parameters
        ----------------
        **kwargs:
            Other keyword arguments are:

            - **DataLoader related parameters**:<br/>
                For example,  `sampler`, `batch_sampler`, `collate_fn`, etc.
                Please, see the pytorch
                [DataLoader docs](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader)
                for details.

            - **Label Distribution Smoothing related parameters**:<br/>

                - lds_kernel (`Literal['gaussian', 'triang', 'laplace']`):
                    choice of kernel for Label Distribution Smoothing
                - lds_ks (`int`):
                    LDS kernel window size
                - lds_sigma (`float`):
                    standard deviation of ['gaussian','laplace'] kernel for LDS
                - lds_granularity (`int`):
                    number of bins in histogram used in LDS to count occurence of sample values
                - lds_reweight (`bool`):
                    option to reweight bin frequency counts in LDS
                - lds_y_max (`Optional[float]`):
                    option to restrict LDS bins by upper label limit
                - lds_y_min (`Optional[float]`):
                    option to restrict LDS bins by lower label limit

                See `pytorch_widedeep.trainer._wd_dataset` for more details on
                the implications of these parameters

            - **Finetune related parameters**:<br/>
                see the source code at `pytorch_widedeep._finetune`. Namely, these are:

                - `finetune_epochs` (`int`):
                    number of epochs use for fine tuning
                - `finetune_max_lr` (`float`):
                   max lr during fine tuning
                - `routine` (`str`):
                   one of _'howard'_ or _'felbo'_
                - `deeptabular_gradual` (`bool`):
                   boolean indicating if the `deeptabular` component will be fine tuned gradually
                - `deeptabular_layers` (`Optional[Union[List[nn.Module], List[List[nn.Module]]]]`):
                   List of pytorch modules indicating the layers of the
                   `deeptabular` that will be fine tuned
                - `deeptabular_max_lr` (`Union[float, List[float]]`):
                   max lr for the `deeptabular` componet during fine tuning
                - `deeptext_gradual` (`bool`):
                   same as `deeptabular_gradual` but for the `deeptext` component
                - `deeptext_layers` (`Optional[Union[List[nn.Module], List[List[nn.Module]]]]`):
                   same as `deeptabular_gradual` but for the `deeptext` component.
                   If there are multiple text columns/models, this should be a list of lists
                - `deeptext_max_lr` (`Union[float, List[float]]`):
                   same as `deeptabular_gradual` but for the `deeptext` component
                   If there are multiple text columns/models, this should be a list of floats
                - `deepimage_gradual` (`bool`):
                   same as `deeptext_layers` but for the `deepimage` component
                - `deepimage_layers` (`Optional[Union[List[nn.Module], List[List[nn.Module]]]]`):
                   same as `deeptext_layers` but for the `deepimage` component
                - `deepimage_max_lr` (`Union[float, List[float]]`):
                    same as `deeptext_layers` but for the `deepimage` component

        Examples
        --------

        For a series of comprehensive examples on how to use the `fit` method, please see the
        [Examples](https://github.com/jrzaurin/pytorch-widedeep/tree/master/examples)
        folder in the repo
        """

        lds_args, dataloader_args, finetune_args = self._extract_kwargs(kwargs)
        lds_args["with_lds"] = with_lds
        self.with_lds = with_lds

        self.batch_size = batch_size

        train_set, eval_set = wd_train_val_split(
            self.seed,
            self.method,  # type: ignore
            X_wide,
            X_tab,
            X_text,
            X_img,
            X_train,
            X_val,
            val_split,
            target,
            self.transforms,
            **lds_args,
        )
        if isinstance(custom_dataloader, type):
            if issubclass(custom_dataloader, DataLoader):
                train_loader = custom_dataloader(  # type: ignore[misc]
                    dataset=train_set,
                    batch_size=batch_size,
                    num_workers=self.num_workers,
                    **dataloader_args,
                )
            else:
                NotImplementedError(
                    "Custom DataLoader must be a subclass of "
                    "torch.utils.data.DataLoader, please see the "
                    "pytorch documentation or examples in "
                    "pytorch_widedeep.dataloaders"
                )
        else:
            train_loader = DataLoaderDefault(
                dataset=train_set,
                batch_size=batch_size,
                num_workers=self.num_workers,
                **dataloader_args,
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

        if finetune:
            self.with_finetuning: bool = True
            self._finetune(train_loader, **finetune_args)
            if self.verbose:
                print(
                    "Fine-tuning (or warmup) of individual components completed. "
                    "Training the whole model for {} epochs".format(n_epochs)
                )
        else:
            self.with_finetuning = False

        self.callback_container.on_train_begin(
            {"batch_size": batch_size, "train_steps": train_steps, "n_epochs": n_epochs}
        )
        for epoch in range(n_epochs):
            epoch_logs: Dict[str, float] = {}
            self.callback_container.on_epoch_begin(epoch, logs=epoch_logs)

            self.train_running_loss = 0.0
            with trange(train_steps, disable=self.verbose != 1) as t:
                for batch_idx, (data, targett, lds_weightt) in zip(t, train_loader):
                    t.set_description("epoch %i" % (epoch + 1))
                    train_score, train_loss = self._train_step(
                        data, targett, batch_idx, epoch, lds_weightt
                    )
                    print_loss_and_metric(t, train_loss, train_score)
                    self.callback_container.on_batch_end(batch=batch_idx)
            epoch_logs = save_epoch_logs(epoch_logs, train_loss, train_score, "train")

            on_epoch_end_metric = None
            if eval_set is not None and epoch % validation_freq == (
                validation_freq - 1
            ):
                self.callback_container.on_eval_begin()
                self.valid_running_loss = 0.0
                with trange(eval_steps, disable=self.verbose != 1) as v:
                    for i, (data, targett) in zip(v, eval_loader):
                        v.set_description("valid")
                        val_score, val_loss = self._eval_step(data, targett, i)
                        print_loss_and_metric(v, val_loss, val_score)
                epoch_logs = save_epoch_logs(epoch_logs, val_loss, val_score, "val")

                if self.reducelronplateau:
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

            if self.model.with_fds:
                self._update_fds_stats(train_loader, epoch)

        self.callback_container.on_train_end(epoch_logs)

        if feature_importance_sample_size is not None:
            self.feature_importance = FeatureImportance(
                self.device, feature_importance_sample_size
            ).feature_importance(train_loader, self.model)
        self._restore_best_weights()
        self.model.train()

    def predict(  # type: ignore[override, return]
        self,
        X_wide: Optional[np.ndarray] = None,
        X_tab: Optional[np.ndarray] = None,
        X_text: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
        X_img: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
        X_test: Optional[Dict[str, Union[np.ndarray, List[np.ndarray]]]] = None,
        batch_size: Optional[int] = None,
    ) -> np.ndarray:
        r"""Returns the predictions

        The input datasets can be passed either directly via numpy arrays
        (`X_wide`, `X_tab`, `X_text` or `X_img`) or alternatively, in
        a dictionary (`X_test`)


        Parameters
        ----------
        X_wide: np.ndarray, Optional. default=None
            Input for the `wide` model component.
            See `pytorch_widedeep.preprocessing.WidePreprocessor`
        X_tab: np.ndarray, Optional. default=None
            Input for the `deeptabular` model component.
            See `pytorch_widedeep.preprocessing.TabPreprocessor`
        X_text: np.ndarray, Optional. default=None
            Input for the `deeptext` model component.
            See `pytorch_widedeep.preprocessing.TextPreprocessor`
        X_img: np.ndarray, Optional. default=None
            Input for the `deepimage` model component.
            See `pytorch_widedeep.preprocessing.ImagePreprocessor`
        X_test: Dict, Optional. default=None
            The test dataset can also be passed in a dictionary. Keys are
            `X_wide`, _'X_tab'_, _'X_text'_, _'X_img'_ and _'target'_. Values
            are the corresponding matrices.
        batch_size: int, default = 256
            If a trainer is used to predict after having trained a model, the
            `batch_size` needs to be defined as it will not be defined as
            the `Trainer` is instantiated

        Returns
        -------
        np.ndarray:
            array with the predictions
        """
        preds_l = self._predict(X_wide, X_tab, X_text, X_img, X_test, batch_size)
        if self.method == "regression":
            return np.vstack(preds_l).squeeze(1)
        if self.method == "binary":
            preds = np.vstack(preds_l).squeeze(1)
            return (preds > 0.5).astype("int")
        if self.method == "qregression":
            return np.vstack(preds_l)
        if self.method == "multiclass":
            preds = np.vstack(preds_l)
            return np.argmax(preds, 1)  # type: ignore[return-value]

    def predict_uncertainty(  # type: ignore[return]
        self,
        X_wide: Optional[np.ndarray] = None,
        X_tab: Optional[np.ndarray] = None,
        X_text: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
        X_img: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
        X_test: Optional[Dict[str, Union[np.ndarray, List[np.ndarray]]]] = None,
        batch_size: Optional[int] = None,
        uncertainty_granularity=1000,
    ) -> np.ndarray:
        r"""Returns the predicted ucnertainty of the model for the test dataset
        using a Monte Carlo method during which dropout layers are activated
        in the evaluation/prediction phase and each sample is predicted N
        times (`uncertainty_granularity` times).

        This is based on
        [Dropout as a Bayesian Approximation: Representing
        Model Uncertainty in Deep Learning](https://arxiv.org/abs/1506.02142?context=stat).

        Parameters
        ----------
        X_wide: np.ndarray, Optional. default=None
            Input for the `wide` model component.
            See `pytorch_widedeep.preprocessing.WidePreprocessor`
        X_tab: np.ndarray, Optional. default=None
            Input for the `deeptabular` model component.
            See `pytorch_widedeep.preprocessing.TabPreprocessor`
        X_text: np.ndarray, Optional. default=None
            Input for the `deeptext` model component.
            See `pytorch_widedeep.preprocessing.TextPreprocessor`
        X_img: np.ndarray, Optional. default=None
            Input for the `deepimage` model component.
            See `pytorch_widedeep.preprocessing.ImagePreprocessor`
        X_test: Dict, Optional. default=None
            The test dataset can also be passed in a dictionary. Keys are
            _'X_wide'_, _'X_tab'_, _'X_text'_, _'X_img'_ and _'target'_. Values
            are the corresponding matrices.
        batch_size: int, default = 256
            If a trainer is used to predict after having trained a model, the
            `batch_size` needs to be defined as it will not be defined as
            the `Trainer` is instantiated
        uncertainty_granularity: int default = 1000
            number of times the model does prediction for each sample

        Returns
        -------
        np.ndarray:
            - if `method = regression`, it will return an array with `(max, min, mean, stdev)`
              values for each sample.
            - if `method = binary` it will return an array with
              `(mean_cls_0_prob, mean_cls_1_prob, predicted_cls)` for each sample.
            - if `method = multiclass` it will return an array with
              `(mean_cls_0_prob, mean_cls_1_prob, mean_cls_2_prob, ... , predicted_cls)`
              values for each sample.

        """
        preds_l = self._predict(
            X_wide,
            X_tab,
            X_text,
            X_img,
            X_test,
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

    def predict_proba(  # type: ignore[override, return]  # noqa: C901
        self,
        X_wide: Optional[np.ndarray] = None,
        X_tab: Optional[np.ndarray] = None,
        X_text: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
        X_img: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
        X_test: Optional[Dict[str, Union[np.ndarray, List[np.ndarray]]]] = None,
        batch_size: Optional[int] = None,
    ) -> np.ndarray:
        r"""Returns the predicted probabilities for the test dataset for  binary
        and multiclass methods

        The input datasets can be passed either directly via numpy arrays
        (`X_wide`, `X_tab`, `X_text` or `X_img`) or alternatively, in
        a dictionary (`X_test`)

        Parameters
        ----------
        X_wide: np.ndarray, Optional. default=None
            Input for the `wide` model component.
            See `pytorch_widedeep.preprocessing.WidePreprocessor`
        X_tab: np.ndarray, Optional. default=None
            Input for the `deeptabular` model component.
            See `pytorch_widedeep.preprocessing.TabPreprocessor`
        X_text: np.ndarray, Optional. default=None
            Input for the `deeptext` model component.
            See `pytorch_widedeep.preprocessing.TextPreprocessor`
        X_img: np.ndarray, Optional. default=None
            Input for the `deepimage` model component.
            See `pytorch_widedeep.preprocessing.ImagePreprocessor`
        X_test: Dict, Optional. default=None
            The test dataset can also be passed in a dictionary. Keys are
            `X_wide`, _'X_tab'_, _'X_text'_, _'X_img'_ and _'target'_. Values
            are the corresponding matrices.
        batch_size: int, default = 256
            If a trainer is used to predict after having trained a model, the
            `batch_size` needs to be defined as it will not be defined as
            the `Trainer` is instantiated

        Returns
        -------
        np.ndarray
            array with the probabilities per class
        """

        preds_l = self._predict(X_wide, X_tab, X_text, X_img, X_test, batch_size)
        if self.method == "binary":
            preds = np.vstack(preds_l).squeeze(1)
            probs = np.zeros([preds.shape[0], 2])
            probs[:, 0] = 1 - preds
            probs[:, 1] = preds
            return probs
        if self.method == "multiclass":
            return np.vstack(preds_l)

    def explain(self, X_tab: np.ndarray, save_step_masks: Optional[bool] = None):
        # TO DO: Add docs to this, to the feat imp parameter and the all
        # related classes
        explainer = Explainer(self.device)

        res = explainer.explain(
            self.model, X_tab, self.num_workers, self.batch_size, save_step_masks
        )

        return res

    def save(
        self,
        path: str,
        save_state_dict: bool = False,
        model_filename: str = "wd_model.pt",
    ):
        r"""Saves the model, training and evaluation history, and the
        `feature_importance` attribute (if the `deeptabular` component is a
        Tabnet model) to disk

        The `Trainer` class is built so that it 'just' trains a model. With
        that in mind, all the torch related parameters (such as optimizers,
        learning rate schedulers, initializers, etc) have to be defined
        externally and then passed to the `Trainer`. As a result, the
        `Trainer` does not generate any attribute or additional data
        products that need to be saved other than the `model` object itself,
        which can be saved as any other torch model (e.g. `torch.save(model,
        path)`).

        The exception is Tabnet. If the `deeptabular` component is a Tabnet
        model, an attribute (a dict) called `feature_importance` will be
        created at the end of the training process. Therefore, a `save`
        method was created that will save the feature importance dictionary
        to a json file and, since we are here, the model weights, training
        history and learning rate history.

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

        if self.model.is_tabnet:
            with open(save_dir / "feature_importance.json", "w") as fi:
                json.dump(self.feature_importance, fi)

    @alias("n_epochs", ["finetune_epochs", "warmup_epochs"])
    @alias("max_lr", ["finetune_max_lr", "warmup_max_lr"])
    def _finetune(
        self,
        loader: DataLoader,
        n_epochs: int = 5,
        max_lr: float = 0.01,
        routine: Literal["howard", "felbo"] = "howard",
        deeptabular_gradual: bool = False,
        deeptabular_layers: Optional[
            Union[List[nn.Module], List[List[nn.Module]]]
        ] = None,
        deeptabular_max_lr: Union[float, List[float]] = 0.01,
        deeptext_gradual: bool = False,
        deeptext_layers: Optional[Union[List[nn.Module], List[List[nn.Module]]]] = None,
        deeptext_max_lr: Union[float, List[float]] = 0.01,
        deepimage_gradual: bool = False,
        deepimage_layers: Optional[
            Union[List[nn.Module], List[List[nn.Module]]]
        ] = None,
        deepimage_max_lr: Union[float, List[float]] = 0.01,
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
                ), "deeptext_layers must be passed if deeptext_gradual=True"
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
                ), "deepimage_layers must be passed if deepimage_gradual=True"
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
        lds_weightt: Tensor,
    ):
        lds_weight = (
            None
            if torch.all(lds_weightt == 0)
            else lds_weightt.view(-1, 1).to(self.device)
        )
        if (
            self.with_lds
            and lds_weight is not None
            and "lds_weight" not in signature(self.loss_fn.forward).parameters
        ):
            warnings.warn(
                """LDS weights are not None but the loss function used does not"
                " support LDS weightening. For loss functions that support LDS"
                " weightening please read the docs""",
                UserWarning,
            )

        self.model.train()

        X: Dict[str, Union[Tensor, List[Tensor]]] = {}
        for k, v in data.items():
            if isinstance(v, list):
                X[k] = [i.to(self.device) for i in v]
            else:
                X[k] = v.to(self.device)
        y = (
            target.view(-1, 1).float()
            if self.method not in ["multiclass", "qregression", "multitarget"]
            else target
        )
        y = y.to(self.device)

        self.optimizer.zero_grad()

        if self.model.with_fds:
            _, y_pred = self.model(X, y, epoch)
        else:
            y_pred = self.model(X)

        if self.model.is_tabnet:
            loss = self.loss_fn(y_pred[0], y) - self.lambda_sparse * y_pred[1]
            score = self._get_score(y_pred[0], y)
        else:
            loss = (
                self.loss_fn(y_pred, y)
                if not self.with_lds
                else self.loss_fn(y_pred, y, lds_weight=lds_weight)
            )
            score = self._get_score(y_pred, y)

        loss.backward()
        self.optimizer.step()

        self.train_running_loss += loss.item()
        avg_loss = self.train_running_loss / (batch_idx + 1)

        return score, avg_loss

    def _eval_step(
        self,
        data: Dict[str, Union[Tensor, List[Tensor]]],
        target: Tensor,
        batch_idx: int,
    ):
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
            if self.model.is_tabnet:
                loss = self.loss_fn(y_pred[0], y) - self.lambda_sparse * y_pred[1]
                score = self._get_score(y_pred[0], y)
            else:
                score = self._get_score(y_pred, y)
                loss = self.loss_fn(y_pred, y)

            self.valid_running_loss += loss.item()
            avg_loss = self.valid_running_loss / (batch_idx + 1)

        return score, avg_loss

    def _get_score(self, y_pred, y):
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

    def _fds_step(
        self,
        data: Dict[str, Tensor],
        target: Tensor,
        epoch: int,
    ) -> Tuple[Tensor, Tensor]:
        self.model.train()
        # FDS is only supported for the deeptabular component, X will never
        # be Dict[str, List[Tensor]]
        X = {k: v.to(self.device) for k, v in data.items()}
        y = target.view(-1, 1).float().to(self.device)
        smoothed_features, _ = self.model(X, y, epoch)
        return smoothed_features, y

    def _update_fds_stats(self, train_loader: DataLoader, epoch: int):
        train_steps = len(train_loader)
        features_l, y_pred_l = [], []
        with torch.no_grad():
            with trange(train_steps, disable=self.verbose != 1) as t:
                for _, (data, targett, _) in zip(t, train_loader):
                    t.set_description("FDS update")
                    deeptab_features, deeptab_preds = self._fds_step(
                        data,
                        targett,
                        epoch,
                    )
                    features_l.append(deeptab_features)
                    y_pred_l.append(deeptab_preds)
        features = torch.cat(features_l)
        y_pred = torch.cat(y_pred_l)
        self.model.fds_layer.update_last_epoch_stats(epoch)
        self.model.fds_layer.update_running_stats(features, y_pred, epoch)

    def _predict(  # type: ignore[override, return]  # noqa: C901
        self,
        X_wide: Optional[np.ndarray] = None,
        X_tab: Optional[np.ndarray] = None,
        X_text: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
        X_img: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
        X_test: Optional[Dict[str, Union[np.ndarray, List[np.ndarray]]]] = None,
        batch_size: Optional[int] = None,
        uncertainty_granularity=1000,
        uncertainty: bool = False,
    ) -> List:
        r"""Private method to avoid code repetition in predict and
        predict_proba. For parameter information, please, see the .predict()
        method documentation
        """
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

        if uncertainty:
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
    def _predict_ziln(preds: Tensor) -> Tensor:
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
        dataloader_params = [
            "shuffle",
            "sampler",
            "batch_sampler",
            "num_workers",
            "collate_fn",
            "pin_memory",
            "drop_last",
            "timeout",
            "worker_init_fn",
            "generator",
            "prefetch_factor",
            "persistent_workers",
            "oversample_mul",
        ]
        lds_params = [
            "lds_kernel",
            "lds_ks",
            "lds_sigma",
            "lds_granularity",
            "lds_reweight",
            "lds_y_max",
            "lds_y_min",
        ]
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

        lds_args, dataloader_args, finetune_args = {}, {}, {}
        for k, v in kwargs.items():
            if k in lds_params:
                lds_args[k] = v
            if k in dataloader_params:
                dataloader_args[k] = v
            if k in finetune_params:
                finetune_args[k] = v

        return lds_args, dataloader_args, finetune_args
