import os
import sys
import json
import warnings
from inspect import signature
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import trange
from torch import nn
from scipy.sparse import csc_matrix
from torchmetrics import Metric as TorchMetric
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from pytorch_widedeep.losses import ZILNLoss
from pytorch_widedeep.metrics import Metric, MultipleMetrics
from pytorch_widedeep.wdtypes import *  # noqa: F403
from pytorch_widedeep.callbacks import (
    History,
    Callback,
    MetricCallback,
    CallbackContainer,
    LRShedulerCallback,
)
from pytorch_widedeep.dataloaders import DataLoaderDefault
from pytorch_widedeep.initializers import Initializer, MultipleInitializer
from pytorch_widedeep.training._finetune import FineTune
from pytorch_widedeep.utils.general_utils import Alias
from pytorch_widedeep.training._wd_dataset import WideDeepDataset
from pytorch_widedeep.training._trainer_utils import (
    alias_to_loss,
    save_epoch_logs,
    wd_train_val_split,
    print_loss_and_metric,
)
from pytorch_widedeep.models.tabular.tabnet._utils import create_explain_matrix
from pytorch_widedeep.training._multiple_optimizer import MultipleOptimizer
from pytorch_widedeep.training._multiple_transforms import MultipleTransforms
from pytorch_widedeep.training._loss_and_obj_aliases import _ObjectiveToMethod
from pytorch_widedeep.training._multiple_lr_scheduler import (
    MultipleLRScheduler,
)


class Trainer:
    r"""Class to set the of attributes that will be used during the
    training process.

    Parameters
    ----------
    model: ``WideDeep``
        An object of class ``WideDeep``
    objective: str
        Defines the objective, loss or cost function.

        Param aliases: ``loss_function``, ``loss_fn``, ``loss``,
        ``cost_function``, ``cost_fn``, ``cost``

        Possible values are:

        - ``binary``, aliases: ``logistic``, ``binary_logloss``, ``binary_cross_entropy``

        - ``binary_focal_loss``

        - ``multiclass``, aliases: ``multi_logloss``, ``cross_entropy``, ``categorical_cross_entropy``,

        - ``multiclass_focal_loss``

        - ``regression``, aliases: ``mse``, ``l2``, ``mean_squared_error``

        - ``mean_absolute_error``, aliases: ``mae``, ``l1``

        - ``mean_squared_log_error``, aliases: ``msle``

        - ``root_mean_squared_error``, aliases:  ``rmse``

        - ``root_mean_squared_log_error``, aliases: ``rmsle``

        - ``zero_inflated_lognormal``, aliases: ``ziln``

        - ``quantile``

        - ``tweedie``
    custom_loss_function: ``nn.Module``. Optional, default = None
        It is possible to pass a custom loss function. See for example
        :class:`pytorch_widedeep.losses.FocalLoss` for the required structure
        of the object or the `Examples
        <https://github.com/jrzaurin/pytorch-widedeep/tree/master/examples>`__
        folder in the repo. Note that if ``custom_loss_function`` is not
        None, ``objective`` must be 'binary', 'multiclass' or 'regression',
        consistent with the loss function
    optimizers: ``Optimzer`` or dict. Optional, default= None
        - An instance of Pytorch's :obj:`Optimizer` object
          (e.g. :obj:`torch.optim.Adam()`) or
        - a dictionary where there keys are the model components (i.e.
          `'wide'`, `'deeptabular'`, `'deeptext'`, `'deepimage'` and/or `'deephead'`)  and
          the values are the corresponding optimizers. If multiple optimizers are used
          the  dictionary **MUST** contain an optimizer per model component.

        if no optimizers are passed it will default to :obj:`Adam` for all
        model components
    lr_schedulers: ``LRScheduler`` or dict. Optional, default=None
        - An instance of Pytorch's :obj:`LRScheduler` object (e.g
          :obj:`torch.optim.lr_scheduler.StepLR(opt, step_size=5)`) or
        - a dictionary where there keys are the model componenst (i.e. `'wide'`,
          `'deeptabular'`, `'deeptext'`, `'deepimage'` and/or `'deephead'`) and the
          values are the corresponding learning rate schedulers.
    initializers: ``Initializer`` or dict. Optional, default=None
        - An instance of an `Initializer`` object see :obj:`pytorch-widedeep.initializers` or
        - a dictionary where there keys are the model components (i.e. `'wide'`,
          `'deeptabular'`, `'deeptext'`, `'deepimage'` and/or `'deephead'`)
          and the values are the corresponding initializers.
    transforms: List. Optional, default=None
        List with :obj:`torchvision.transforms` to be applied to the image
        component of the model (i.e. ``deepimage``) See `torchvision
        transforms
        <https://pytorch.org/docs/stable/torchvision/transforms.html>`_.
    callbacks: List. Optional, default=None
        List with :obj:`Callback` objects. The three callbacks available in
        :obj:`pytorch-widedeep` are: :obj:`LRHistory`, :obj:`ModelCheckpoint`
        and :obj:`EarlyStopping`. The :obj:`History` and
        the :obj:`LRShedulerCallback` callbacks are used by default. This
        can also be a custom callback as long as the object of
        type :obj:`Callback`. See
        :obj:`pytorch_widedeep.callbacks.Callback` or the examples folder
        in the repo
    metrics: List. Optional, default=None
        - List of objects of type :obj:`Metric`. Metrics available are:
          :obj:`Accuracy`, :obj:`Precision`, :obj:`Recall`, :obj:`FBetaScore`,
          `F1Score` and `R2Score`. This can also be a custom metric as long
          as it is an object of type :obj:`Metric`. See
          :obj:`pytorch_widedeep.metrics.Metric` or the examples folder in the
          repo
        - List of objects of type :obj:`torchmetrics.Metric`. This can be any
          metric from torchmetrics library `Examples
          <https://torchmetrics.readthedocs.io/en/latest/references/modules.html#
          classification-metrics>`_. This can also be a custom metric as long as
          it is an object of type :obj:`Metric`. See `the instructions
          <https://torchmetrics.readthedocs.io/en/latest/>`_.
    verbose: int, default=1
        Verbosity level. If set to 0 nothing will be printed during training
    seed: int, default=1
        Random seed to be used internally for train/test split

    Attributes
    ----------
    cyclic_lr: bool
        Attribute that indicates if any of the lr_schedulers is cyclic_lr
        (i.e. :obj:`CyclicLR` or
        :obj:`OneCycleLR`). See `Pytorch schedulers
        <https://pytorch.org/docs/stable/optim.html>`_.
    feature_importance: dict
        dict where the keys are the column names and the values are the
        corresponding feature importances. This attribute will only exist
        if the ``deeptabular`` component is a Tabnet model

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

    @Alias(  # noqa: C901
        "objective",
        ["loss_function", "loss_fn", "loss", "cost_function", "cost_fn", "cost"],
    )
    def __init__(
        self,
        model: WideDeep,
        objective: str,
        custom_loss_function: Optional[Module] = None,
        optimizers: Optional[Union[Optimizer, Dict[str, Optimizer]]] = None,
        lr_schedulers: Optional[Union[LRScheduler, Dict[str, LRScheduler]]] = None,
        initializers: Optional[Union[Initializer, Dict[str, Initializer]]] = None,
        transforms: Optional[List[Transforms]] = None,
        callbacks: Optional[List[Callback]] = None,
        metrics: Optional[Union[List[Metric], List[TorchMetric]]] = None,
        verbose: int = 1,
        seed: int = 1,
        **kwargs,
    ):

        self._check_inputs(
            model, objective, optimizers, lr_schedulers, custom_loss_function
        )
        self.device, self.num_workers = self._set_device_and_num_workers(**kwargs)

        # initialize early_stop. If EarlyStopping Callback is used it will
        # take care of it
        self.early_stop = False
        self.verbose = verbose
        self.seed = seed

        self.model = model
        if self.model.is_tabnet:
            self.lambda_sparse = kwargs.get("lambda_sparse", 1e-3)
            self.reducing_matrix = create_explain_matrix(self.model)
        self.model.to(self.device)

        self.objective = objective
        self.method = _ObjectiveToMethod.get(objective)

        self._initialize(initializers)
        self.loss_fn = self._set_loss_fn(objective, custom_loss_function, **kwargs)
        self.optimizer = self._set_optimizer(optimizers)
        self.lr_scheduler = self._set_lr_scheduler(lr_schedulers, **kwargs)
        self.transforms = self._set_transforms(transforms)
        self._set_callbacks_and_metrics(callbacks, metrics)

    @Alias("finetune", "warmup")  # noqa: C901
    def fit(
        self,
        X_wide: Optional[np.ndarray] = None,
        X_tab: Optional[np.ndarray] = None,
        X_text: Optional[np.ndarray] = None,
        X_img: Optional[np.ndarray] = None,
        X_train: Optional[Dict[str, np.ndarray]] = None,
        X_val: Optional[Dict[str, np.ndarray]] = None,
        val_split: Optional[float] = None,
        target: Optional[np.ndarray] = None,
        n_epochs: int = 1,
        validation_freq: int = 1,
        batch_size: int = 32,
        custom_dataloader: Optional[DataLoader] = None,
        finetune: bool = False,
        with_lds: bool = False,
        **kwargs,
    ):
        r"""Fit method.

        The input datasets can be passed either directly via numpy arrays
        (``X_wide``, ``X_tab``, ``X_text`` or ``X_img``) or alternatively, in
        dictionaries (``X_train`` or ``X_val``).

        Parameters
        ----------
        X_wide: np.ndarray, Optional. default=None
            Input for the ``wide`` model component.
            See :class:`pytorch_widedeep.preprocessing.WidePreprocessor`
        X_tab: np.ndarray, Optional. default=None
            Input for the ``deeptabular`` model component.
            See :class:`pytorch_widedeep.preprocessing.TabPreprocessor`
        X_text: np.ndarray, Optional. default=None
            Input for the ``deeptext`` model component.
            See :class:`pytorch_widedeep.preprocessing.TextPreprocessor`
        X_img : np.ndarray, Optional. default=None
            Input for the ``deepimage`` model component.
            See :class:`pytorch_widedeep.preprocessing.ImagePreprocessor`
        X_train: Dict, Optional. default=None
            The training dataset can also be passed in a dictionary. Keys are
            `X_wide`, `'X_tab'`, `'X_text'`, `'X_img'` and `'target'`. Values
            are the corresponding matrices.
        X_val: Dict, Optional. default=None
            The validation dataset can also be passed in a dictionary. Keys
            are `X_wide`, `'X_tab'`, `'X_text'`, `'X_img'` and `'target'`.
            Values are the corresponding matrices.
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
        custom_dataloader: ``DataLoader``, Optional, default=None
            object of class ``torch.utils.data.DataLoader``. Available
            predefined dataloaders are in ``pytorch-widedeep.dataloaders``.If
            ``None``, a standard torch ``DataLoader`` is used.
        finetune: bool, default=False
            param alias: ``warmup``

            fine-tune individual model components. This functionality can also
            be used to 'warm-up' individual components before the joined
            training starts, and hence its alias. See the Examples folder in
            the repo for more details

            ``pytorch_widedeep`` implements 3 fine-tune routines.

            - fine-tune all trainable layers at once. This routine is is
              inspired by the work of Howard & Sebastian Ruder 2018 in their
              `ULMfit paper <https://arxiv.org/abs/1801.06146>`_. Using a
              Slanted Triangular learing (see `Leslie N. Smith paper
              <https://arxiv.org/pdf/1506.01186.pdf>`_), the process is the
              following: `i`) the learning rate will gradually increase for
              10% of the training steps from max_lr/10 to max_lr. `ii`) It
              will then gradually decrease to max_lr/10 for the remaining 90%
              of the steps. The optimizer used in the process is ``Adam``.

            and two gradual fine-tune routines, where only certain layers are
            trained at a time.

            - The so called `Felbo` gradual fine-tune rourine, based on the the
              Felbo et al., 2017 `DeepEmoji paper <https://arxiv.org/abs/1708.00524>`_.
            - The `Howard` routine based on the work of Howard & Sebastian Ruder 2018 in their
              `ULMfit paper <https://arxiv.org/abs/1801.06146>`_.

            For details on how these routines work, please see the Examples
            section in this documentation and the `Examples
            <https://github.com/jrzaurin/pytorch-widedeep/tree/master/examples>`__
            folder in the repo.

        Examples
        --------

        For a series of comprehensive examples please, see the `Examples
        <https://github.com/jrzaurin/pytorch-widedeep/tree/master/examples>`__
        folder in the repo

        For completion, here we include some `"fabricated"` examples, i.e.
        these assume you have already built a model and instantiated a
        ``Trainer``, that is ready to fit

        .. code-block:: python

            # Ex 1. using train input arrays directly and no validation
            trainer.fit(X_wide=X_wide, X_tab=X_tab, target=target, n_epochs=10, batch_size=256)


        .. code-block:: python

            # Ex 2: using train input arrays directly and validation with val_split
            trainer.fit(X_wide=X_wide, X_tab=X_tab, target=target, n_epochs=10, batch_size=256, val_split=0.2)


        .. code-block:: python

            # Ex 3: using train dict and val_split
            X_train = {'X_wide': X_wide, 'X_tab': X_tab, 'target': y}
            trainer.fit(X_train, n_epochs=10, batch_size=256, val_split=0.2)


        .. code-block:: python

            # Ex 4: validation using training and validation dicts
            X_train = {'X_wide': X_wide_tr, 'X_tab': X_tab_tr, 'target': y_tr}
            X_val = {'X_wide': X_wide_val, 'X_tab': X_tab_val, 'target': y_val}
            trainer.fit(X_train=X_train, X_val=X_val n_epochs=10, batch_size=256)
        """

        lds_args, dataloader_args, finetune_args = self._extract_kwargs(kwargs)
        lds_args["with_lds"] = with_lds
        self.with_lds = with_lds

        self.batch_size = batch_size
        train_set, eval_set = wd_train_val_split(
            self.seed,
            self.method,
            X_wide,
            X_tab,
            X_text,
            X_img,
            X_train,
            X_val,
            val_split,
            target,
            **lds_args,
        )
        if isinstance(custom_dataloader, type):
            if issubclass(custom_dataloader, DataLoader):
                train_loader = custom_dataloader(
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
            self._finetune(train_loader, **finetune_args)
            if self.verbose:
                print(
                    "Fine-tuning (or warmup) of individual components completed. "
                    "Training the whole model for {} epochs".format(n_epochs)
                )

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
                self.callback_container.on_train_end(epoch_logs)
                break

            if self.model.with_fds:
                self._update_fds_stats(train_loader, epoch)

        self.callback_container.on_train_end(epoch_logs)
        if self.model.is_tabnet:
            self._compute_feature_importance(train_loader)
        self._restore_best_weights()
        self.model.train()

    def predict(  # type: ignore[return]
        self,
        X_wide: Optional[np.ndarray] = None,
        X_tab: Optional[np.ndarray] = None,
        X_text: Optional[np.ndarray] = None,
        X_img: Optional[np.ndarray] = None,
        X_test: Optional[Dict[str, np.ndarray]] = None,
        batch_size: int = 256,
    ) -> np.ndarray:
        r"""Returns the predictions

        The input datasets can be passed either directly via numpy arrays
        (``X_wide``, ``X_tab``, ``X_text`` or ``X_img``) or alternatively, in
        a dictionary (``X_test``)


        Parameters
        ----------
        X_wide: np.ndarray, Optional. default=None
            Input for the ``wide`` model component.
            See :class:`pytorch_widedeep.preprocessing.WidePreprocessor`
        X_tab: np.ndarray, Optional. default=None
            Input for the ``deeptabular`` model component.
            See :class:`pytorch_widedeep.preprocessing.TabPreprocessor`
        X_text: np.ndarray, Optional. default=None
            Input for the ``deeptext`` model component.
            See :class:`pytorch_widedeep.preprocessing.TextPreprocessor`
        X_img : np.ndarray, Optional. default=None
            Input for the ``deepimage`` model component.
            See :class:`pytorch_widedeep.preprocessing.ImagePreprocessor`
        X_test: Dict, Optional. default=None
            The test dataset can also be passed in a dictionary. Keys are
            `X_wide`, `'X_tab'`, `'X_text'`, `'X_img'` and `'target'`. Values
            are the corresponding matrices.
        batch_size: int, default = 256
            If a trainer is used to predict after having trained a model, the
            ``batch_size`` needs to be defined as it will not be defined as
            the :obj:`Trainer` is instantiated
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
        X_text: Optional[np.ndarray] = None,
        X_img: Optional[np.ndarray] = None,
        X_test: Optional[Dict[str, np.ndarray]] = None,
        batch_size: int = 256,
        uncertainty_granularity=1000,
    ) -> np.ndarray:
        r"""Returns the predicted ucnertainty of the model for the test dataset
        using a Monte Carlo method during which dropout layers are activated
        in the evaluation/prediction phase and each sample is predicted N
        times (``uncertainty_granularity`` times).

        This is based on `'Gal Y. & Ghahramani Z., 2016, Dropout as a Bayesian
        Approximation: Representing Model Uncertainty in Deep Learning'`,
        Proceedings of the 33rd International Conference on Machine Learning

        Parameters
        ----------
        X_wide: np.ndarray, Optional. default=None
            Input for the ``wide`` model component.
            See :class:`pytorch_widedeep.preprocessing.WidePreprocessor`
        X_tab: np.ndarray, Optional. default=None
            Input for the ``deeptabular`` model component.
            See :class:`pytorch_widedeep.preprocessing.TabPreprocessor`
        X_text: np.ndarray, Optional. default=None
            Input for the ``deeptext`` model component.
            See :class:`pytorch_widedeep.preprocessing.TextPreprocessor`
        X_img : np.ndarray, Optional. default=None
            Input for the ``deepimage`` model component.
            See :class:`pytorch_widedeep.preprocessing.ImagePreprocessor`
        X_test: Dict, Optional. default=None
            The test dataset can also be passed in a dictionary. Keys are
            `X_wide`, `'X_tab'`, `'X_text'`, `'X_img'` and `'target'`. Values
            are the corresponding matrices.
        batch_size: int, default = 256
            If a trainer is used to predict after having trained a model, the
            ``batch_size`` needs to be defined as it will not be defined as
            the :obj:`Trainer` is instantiated
        uncertainty_granularity: int default = 1000
            number of times the model does prediction for each sample if uncertainty
            is set to True

        Returns
        -------
            - if ``method == regression``, it will return an array with `{max, min, mean, stdev}`
              values for each sample.
            - if ``method == binary`` it will return an array with
              `{mean_cls_0_prob, mean_cls_1_prob, predicted_cls}` for each sample.
            - if ``method == multiclass`` it will return an array with
              `{mean_cls_0_prob, mean_cls_1_prob, mean_cls_2_prob, ... , predicted_cls}` values for each sample.

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

    def predict_proba(  # type: ignore[return]
        self,
        X_wide: Optional[np.ndarray] = None,
        X_tab: Optional[np.ndarray] = None,
        X_text: Optional[np.ndarray] = None,
        X_img: Optional[np.ndarray] = None,
        X_test: Optional[Dict[str, np.ndarray]] = None,
        batch_size: int = 256,
    ) -> np.ndarray:
        r"""Returns the predicted probabilities for the test dataset for  binary
        and multiclass methods

        The input datasets can be passed either directly via numpy arrays
        (``X_wide``, ``X_tab``, ``X_text`` or ``X_img``) or alternatively, in
        a dictionary (``X_test``)

        Parameters
        ----------
        X_wide: np.ndarray, Optional. default=None
            Input for the ``wide`` model component.
            See :class:`pytorch_widedeep.preprocessing.WidePreprocessor`
        X_tab: np.ndarray, Optional. default=None
            Input for the ``deeptabular`` model component.
            See :class:`pytorch_widedeep.preprocessing.TabPreprocessor`
        X_text: np.ndarray, Optional. default=None
            Input for the ``deeptext`` model component.
            See :class:`pytorch_widedeep.preprocessing.TextPreprocessor`
        X_img : np.ndarray, Optional. default=None
            Input for the ``deepimage`` model component.
            See :class:`pytorch_widedeep.preprocessing.ImagePreprocessor`
        X_test: Dict, Optional. default=None
            The test dataset can also be passed in a dictionary. Keys are
            `X_wide`, `'X_tab'`, `'X_text'`, `'X_img'` and `'target'`. Values
            are the corresponding matrices.
        batch_size: int, default = 256
            If a trainer is used to predict after having trained a model, the
            ``batch_size`` needs to be defined as it will not be defined as
            the :obj:`Trainer` is instantiated
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

    def get_embeddings(
        self, col_name: str, cat_encoding_dict: Dict[str, Dict[str, int]]
    ) -> Dict[str, np.ndarray]:  # pragma: no cover
        r"""Returns the learned embeddings for the categorical features passed through
        ``deeptabular``.

        .. note:: This function will be deprecated in the next relase. Please consider
            using ``Tab2Vec`` instead.

        This method is designed to take an encoding dictionary in the same
        format as that of the :obj:`LabelEncoder` Attribute in the class
        :obj:`TabPreprocessor`. See
        :class:`pytorch_widedeep.preprocessing.TabPreprocessor` and
        :class:`pytorch_widedeep.utils.dense_utils.LabelEncder`.

        Parameters
        ----------
        col_name: str,
            Column name of the feature we want to get the embeddings for
        cat_encoding_dict: Dict
            Dictionary where the keys are the name of the column for which we
            want to retrieve the embeddings and the values are also of type
            Dict. These Dict values have keys that are the categories for that
            column and the values are the corresponding numberical encodings

            e.g.: {'column': {'cat_0': 1, 'cat_1': 2, ...}}

        Examples
        --------

        For a series of comprehensive examples please, see the `Examples
        <https://github.com/jrzaurin/pytorch-widedeep/tree/master/examples>`__
        folder in the repo

        For completion, here we include a `"fabricated"` example, i.e.
        assuming we have already trained the model, that we have the
        categorical encodings in a dictionary name ``encoding_dict``, and that
        there is a column called `'education'`:

        .. code-block:: python

            trainer.get_embeddings(col_name="education", cat_encoding_dict=encoding_dict)
        """
        warnings.warn(
            "'get_embeddings' will be deprecated in the next release. "
            "Please consider using 'Tab2vec' instead",
            DeprecationWarning,
        )
        for n, p in self.model.named_parameters():
            if "embed_layers" in n and col_name in n:
                embed_mtx = p.cpu().data.numpy()
        encoding_dict = cat_encoding_dict[col_name]
        inv_encoding_dict = {v: k for k, v in encoding_dict.items()}
        cat_embed_dict = {}
        for idx, value in inv_encoding_dict.items():
            cat_embed_dict[value] = embed_mtx[idx]
        return cat_embed_dict

    def explain(self, X_tab: np.ndarray, save_step_masks: bool = False):
        """
        if the ``deeptabular`` component is a :obj:`Tabnet` model, returns the
        aggregated feature importance for each instance (or observation) in
        the ``X_tab`` array. If ``save_step_masks`` is set to ``True``, the
        masks per step will also be returned.

        Parameters
        ----------
        X_tab: np.ndarray
            Input array corresponding **only** to the deeptabular component
        save_step_masks: bool
            Boolean indicating if the masks per step will be returned

        Returns
        -------
        res: np.ndarray, Tuple
            Array or Tuple of two arrays with the corresponding aggregated
            feature importance and the masks per step if ``save_step_masks``
            is set to ``True``
        """
        loader = DataLoader(
            dataset=WideDeepDataset(X_tab=X_tab),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

        self.model.eval()
        tabnet_backbone = list(self.model.deeptabular.children())[0]

        m_explain_l = []
        for batch_nb, data in enumerate(loader):
            X = data["deeptabular"].to(self.device)
            M_explain, masks = tabnet_backbone.forward_masks(X)  # type: ignore[operator]
            m_explain_l.append(
                csc_matrix.dot(M_explain.cpu().detach().numpy(), self.reducing_matrix)
            )
            if save_step_masks:
                for key, value in masks.items():
                    masks[key] = csc_matrix.dot(
                        value.cpu().detach().numpy(), self.reducing_matrix
                    )
                if batch_nb == 0:
                    m_explain_step = masks
                else:
                    for key, value in masks.items():
                        m_explain_step[key] = np.vstack([m_explain_step[key], value])

        m_explain_agg = np.vstack(m_explain_l)
        m_explain_agg_norm = m_explain_agg / m_explain_agg.sum(axis=1)[:, np.newaxis]

        res = (
            (m_explain_agg_norm, m_explain_step)
            if save_step_masks
            else np.vstack(m_explain_agg_norm)
        )

        return res

    def save(
        self,
        path: str,
        save_state_dict: bool = False,
        model_filename: str = "wd_model.pt",
    ):
        r"""Saves the model, training and evaluation history, and the
        ``feature_importance`` attribute (if the ``deeptabular`` component is a
        Tabnet model) to disk

        The ``Trainer`` class is built so that it 'just' trains a model. With
        that in mind, all the torch related parameters (such as optimizers,
        learning rate schedulers, initializers, etc) have to be defined
        externally and then passed to the ``Trainer``. As a result, the
        ``Trainer`` does not generate any attribute or additional data
        products that need to be saved other than the ``model`` object itself,
        which can be saved as any other torch model (e.g. ``torch.save(model,
        path)``).

        The exception is Tabnet. If the ``deeptabular`` component is a Tabnet
        model, an attribute (a dict) called ``feature_importance`` will be
        created at the end of the training process. Therefore, a ``save``
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
                        self.model.load_state_dict(callback.best_state_dict)
                    else:
                        if self.verbose:
                            print(
                                "Model weights after training corresponds to the those of the "
                                "final epoch which might not be the best performing weights. Use"
                                "the 'ModelCheckpoint' Callback to restore the best epoch weights."
                            )

    @Alias("n_epochs", ["finetune_epochs", "warmup_epochs"])
    @Alias("max_lr", ["finetune_max_lr", "warmup_max_lr"])
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

        finetuner = FineTune(self.loss_fn, self.metric, self.method, self.verbose)
        if self.model.wide:
            finetuner.finetune_all(self.model.wide, "wide", loader, n_epochs, max_lr)

        if self.model.deeptabular:
            if deeptabular_gradual:
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
        data: Dict[str, Tensor],
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
        X = {k: v.to(self.device) for k, v in data.items()}
        y = (
            target.view(-1, 1).float()
            if self.method not in ["multiclass", "qregression"]
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

    def _eval_step(self, data: Dict[str, Tensor], target: Tensor, batch_idx: int):

        self.model.eval()
        with torch.no_grad():
            X = {k: v.to(self.device) for k, v in data.items()}
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
        X = {k: v.to(self.device) for k, v in data.items()}
        y = target.view(-1, 1).float().to(self.device)
        smoothed_features, _ = self.model(X, y, epoch)
        return smoothed_features, y

    def _update_fds_stats(self, train_loader: DataLoader, epoch: int):
        train_steps = len(train_loader)
        features_l, y_pred_l = [], []
        with torch.no_grad():
            with trange(train_steps, disable=self.verbose != 1) as t:
                for idx, (data, targett, lds_weight) in zip(t, train_loader):
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

    def _compute_feature_importance(self, loader: DataLoader):
        self.model.eval()
        tabnet_backbone = list(self.model.deeptabular.children())[0]
        feat_imp = np.zeros((tabnet_backbone.embed_out_dim))  # type: ignore[arg-type]
        for data, target, _ in loader:
            X = data["deeptabular"].to(self.device)
            y = (
                target.view(-1, 1).float()
                if self.method not in ["multiclass", "qregression"]
                else target
            )
            y = y.to(self.device)
            M_explain, masks = tabnet_backbone.forward_masks(X)  # type: ignore[operator]
            feat_imp += M_explain.sum(dim=0).cpu().detach().numpy()

        feat_imp = csc_matrix.dot(feat_imp, self.reducing_matrix)
        feat_imp = feat_imp / np.sum(feat_imp)

        self.feature_importance = {
            k: v for k, v in zip(tabnet_backbone.column_idx.keys(), feat_imp)  # type: ignore[operator, union-attr]
        }

    def _predict(  # noqa: C901
        self,
        X_wide: Optional[np.ndarray] = None,
        X_tab: Optional[np.ndarray] = None,
        X_text: Optional[np.ndarray] = None,
        X_img: Optional[np.ndarray] = None,
        X_test: Optional[Dict[str, np.ndarray]] = None,
        batch_size: int = 256,
        uncertainty_granularity=1000,
        uncertainty: bool = False,
        quantiles: bool = False,
    ) -> List:
        r"""Private method to avoid code repetition in predict and
        predict_proba. For parameter information, please, see the .predict()
        method documentation
        """
        if X_test is not None:
            test_set = WideDeepDataset(**X_test)  # type: ignore[arg-type]
        else:
            load_dict = {}
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
                for i, k in zip(t, range(prediction_iters)):
                    t.set_description("predict_UncertaintyIter")

                    with trange(
                        test_steps, disable=self.verbose != 1 or uncertainty is True
                    ) as tt:
                        for j, data in zip(tt, test_loader):

                            tt.set_description("predict")
                            X = {k: v.to(self.device) for k, v in data.items()}
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

    def _initialize(self, initializers):
        if initializers is not None:
            if isinstance(initializers, Dict):
                self.initializer = MultipleInitializer(
                    initializers, verbose=self.verbose
                )
                self.initializer.apply(self.model)
            elif isinstance(initializers, type):
                self.initializer = initializers()
                self.initializer(self.model)
            elif isinstance(initializers, Initializer):
                self.initializer = initializers
                self.initializer(self.model)

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

    def _set_optimizer(self, optimizers):
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
                    mod_names.remove("enf_pos")
                    mod_names.remove("fds_layer")
                    optimizers["deeptabular"].add_param_group(
                        {"params": self.model.fds_layer.pred_layer.parameters()}
                    )
                for mn in mod_names:
                    assert mn in opt_names, "No optimizer found for {}".format(mn)
                optimizer = MultipleOptimizer(optimizers)
        else:
            optimizer = torch.optim.Adam(self.model.parameters())  # type: ignore
        return optimizer

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

    def _set_lr_scheduler(self, lr_schedulers, **kwargs):

        # ReduceLROnPlateau is special
        reducelronplateau_criterion = kwargs.get("reducelronplateau_criterion", None)

        self._set_reduce_on_plateau_criterion(
            lr_schedulers, reducelronplateau_criterion
        )

        if lr_schedulers is not None:

            if isinstance(lr_schedulers, LRScheduler) or isinstance(
                lr_schedulers, ReduceLROnPlateau
            ):
                lr_scheduler = lr_schedulers
                cyclic_lr = "cycl" in lr_scheduler.__class__.__name__.lower()
            else:
                lr_scheduler = MultipleLRScheduler(lr_schedulers)
                scheduler_names = [
                    sc.__class__.__name__.lower()
                    for _, sc in lr_scheduler._schedulers.items()
                ]
                cyclic_lr = any(["cycl" in sn for sn in scheduler_names])
        else:
            lr_scheduler, cyclic_lr = None, False

        self.cyclic_lr = cyclic_lr

        return lr_scheduler

    @staticmethod
    def _set_transforms(transforms):
        if transforms is not None:
            return MultipleTransforms(transforms)()
        else:
            return None

    def _set_callbacks_and_metrics(self, callbacks, metrics):
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
        ]:
            raise ValueError(
                "If 'custom_loss_function' is not None, 'objective' must be 'binary' "
                "'multiclass' or 'regression', consistent with the loss function"
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
        default_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device = kwargs.get("device", default_device)
        num_workers = kwargs.get("num_workers", default_num_workers)
        return device, num_workers
