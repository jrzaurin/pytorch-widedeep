import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import trange
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from pytorch_widedeep.losses import MSLELoss, RMSELoss, FocalLoss, RMSLELoss
from pytorch_widedeep.models import WideDeep
from pytorch_widedeep.metrics import Metric, MetricCallback, MultipleMetrics
from pytorch_widedeep.wdtypes import *  # noqa: F403
from pytorch_widedeep.callbacks import History, Callback, CallbackContainer
from pytorch_widedeep.initializers import Initializer, MultipleInitializer
from pytorch_widedeep.training._finetune import FineTune
from pytorch_widedeep.utils.general_utils import Alias
from pytorch_widedeep.training._wd_dataset import WideDeepDataset
from pytorch_widedeep.training._multiple_optimizer import MultipleOptimizer
from pytorch_widedeep.training._multiple_transforms import MultipleTransforms
from pytorch_widedeep.training._loss_and_obj_aliases import (
    _LossAliases,
    _ObjectiveToMethod,
)
from pytorch_widedeep.training._multiple_lr_scheduler import (
    MultipleLRScheduler,
)

n_cpus = os.cpu_count()

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


class Trainer:
    @Alias(  # noqa: C901
        "objective",
        ["loss_function", "loss_fn", "loss", "cost_function", "cost_fn", "cost"],
    )
    @Alias("model", ["model_path"])
    def __init__(
        self,
        model: Union[str, WideDeep],
        objective: str,
        custom_loss_function: Optional[Module] = None,
        optimizers: Optional[Union[Optimizer, Dict[str, Optimizer]]] = None,
        lr_schedulers: Optional[Union[LRScheduler, Dict[str, LRScheduler]]] = None,
        initializers: Optional[Union[Initializer, Dict[str, Initializer]]] = None,
        transforms: Optional[List[Transforms]] = None,
        callbacks: Optional[List[Callback]] = None,
        metrics: Optional[List[Metric]] = None,
        class_weight: Optional[Union[float, List[float], Tuple[float]]] = None,
        alpha: float = 0.25,
        gamma: float = 2,
        verbose: int = 1,
        seed: int = 1,
    ):
        r"""Method to set the of attributes that will be used during the
        training process.

        Parameters
        ----------
        model: str or ``WideDeep``
            param alias: ``model_path``

            An object of class ``WideDeep`` or a string with the full path to
            the model, in which case you might want to use the alias
            ``model_path``.

        objective: str
            Defines the objective, loss or cost function.

            param aliases: ``loss_function``, ``loss_fn``, ``loss``,
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
        custom_loss_function: ``nn.Module``, Optional, default = None
            object of class ``nn.Module``. If none of the loss functions
            available suits the user, it is possible to pass a custom loss
            function. See for example
            :class:`pytorch_widedeep.losses.FocalLoss` for the required
            structure of the object or the `Examples
            <https://github.com/jrzaurin/pytorch-widedeep/tree/master/examples>`_
            folder in the repo.

            .. note:: If ``custom_loss_function`` is not None, ``objective`` must be
                'binary', 'multiclass' or 'regression', consistent with the loss
                function

        optimizers: ``Optimzer`` or Dict, Optional, default= None
            - An instance of Pytorch's ``Optimizer`` object (e.g. :obj:`torch.optim.Adam()`) or
            - a dictionary where there keys are the model components (i.e.
              `'wide'`, `'deeptabular'`, `'deeptext'`, `'deepimage'` and/or `'deephead'`)  and
              the values are the corresponding optimizers. If multiple optimizers are used
              the  dictionary **MUST** contain an optimizer per model component.

            if no optimizers are passed it will default to ``AdamW`` for all
            Wide and Deep components
        lr_schedulers: ``LRScheduler`` or Dict, Optional, default=None
            - An instance of Pytorch's ``LRScheduler`` object (e.g
              :obj:`torch.optim.lr_scheduler.StepLR(opt, step_size=5)`) or
            - a dictionary where there keys are the model componenst (i.e. `'wide'`,
              `'deeptabular'`, `'deeptext'`, `'deepimage'` and/or `'deephead'`) and the
              values are the corresponding learning rate schedulers.
        initializers: ``Initializer`` or Dict, Optional. default=None
            - An instance of an `Initializer`` object see ``pytorch-widedeep.initializers`` or
            - a dictionary where there keys are the model components (i.e. `'wide'`,
              `'deeptabular'`, `'deeptext'`, `'deepimage'` and/or `'deephead'`)
              and the values are the corresponding initializers.
        transforms: List, Optional, default=None
            List with :obj:`torchvision.transforms` to be applied to the image
            component of the model (i.e. ``deepimage``) See `torchvision
            transforms
            <https://pytorch.org/docs/stable/torchvision/transforms.html>`_.
        callbacks: List, Optional, default=None
            List with ``Callback`` objects. The four callbacks available in
            ``pytorch-widedeep`` are: ``History``, ``ModelCheckpoint``,
            ``EarlyStopping``, and ``LRHistory``. The ``History`` callback is
            used by default. This can also be a custom callback as long as the
            object of type ``Callback``. See
            ``pytorch_widedeep.callbacks.Callback`` or the `Examples
            <https://github.com/jrzaurin/pytorch-widedeep/tree/master/examples>`_
            folder in the repo
        metrics: List, Optional, default=None
            List of objects of type ``Metric``. Metrics available are:
            ``Accuracy``, ``Precision``, ``Recall``, ``FBetaScore``,
            ``F1Score`` and ``R2Score``. This can also be a custom metric as
            long as it is an object of type ``Metric``. See
            ``pytorch_widedeep.metrics.Metric`` or the `Examples
            <https://github.com/jrzaurin/pytorch-widedeep/tree/master/examples>`_
            folder in the repo
        class_weight: float, List or Tuple. Optional. default=None
            - float indicating the weight of the minority class in binary classification
              problems (e.g. 9.)
            - a list or tuple with weights for the different classes in multiclass
              classification problems  (e.g. [1., 2., 3.]). The weights do
              not neccesarily need to be normalised. If your loss function
              uses reduction='mean', the loss will be normalized by the sum
              of the corresponding weights for each element. If you are
              using reduction='none', you would have to take care of the
              normalization yourself. See `this discussion
              <https://discuss.pytorch.org/t/passing-the-weights-to-crossentropyloss-correctly/14731/10>`_.
        alpha: float. default=0.25
            if ``objective`` is ``binary_focal_loss`` or
            ``multiclass_focal_loss``, the Focal Loss alpha and gamma
            parameters can be set directly in the ``Trainer`` via the
            ``alpha`` and ``gamma`` parameters
        gamma: float. default=2
            Focal Loss alpha gamma parameter
        verbose: int, default=1
            Setting it to 0 will print nothing during training.
        seed: int, default=1
            Random seed to be used internally for train_test_split

        Attributes
        ----------
        cyclic_lr: bool
            Attribute that indicates if any of the lr_schedulers is cyclic_lr (i.e. ``CyclicLR`` or
            ``OneCycleLR``). See `Pytorch schedulers <https://pytorch.org/docs/stable/optim.html>`_.


        Example
        --------
        >>> import torch
        >>> from torchvision.transforms import ToTensor
        >>>
        >>> # wide deep imports
        >>> from pytorch_widedeep.callbacks import EarlyStopping, LRHistory
        >>> from pytorch_widedeep.initializers import KaimingNormal, KaimingUniform, Normal, Uniform
        >>> from pytorch_widedeep.models import TabResnet, DeepImage, DeepText, Wide, WideDeep
        >>> from pytorch_widedeep import Trainer
        >>> from pytorch_widedeep.optim import RAdam
        >>>
        >>> embed_input = [(u, i, j) for u, i, j in zip(["a", "b", "c"][:4], [4] * 3, [8] * 3)]
        >>> column_idx = {k: v for v, k in enumerate(["a", "b", "c"])}
        >>> wide = Wide(10, 1)
        >>>
        >>> # build the model
        >>> deeptabular = TabResnet(blocks_dims=[8, 4], column_idx=column_idx, embed_input=embed_input)
        >>> deeptext = DeepText(vocab_size=10, embed_dim=4, padding_idx=0)
        >>> deepimage = DeepImage(pretrained=False)
        >>> model = WideDeep(wide=wide, deeptabular=deeptabular, deeptext=deeptext, deepimage=deepimage)
        >>>
        >>> # set optimizers and schedulers
        >>> wide_opt = torch.optim.Adam(model.wide.parameters())
        >>> deep_opt = torch.optim.Adam(model.deeptabular.parameters())
        >>> text_opt = RAdam(model.deeptext.parameters())
        >>> img_opt = RAdam(model.deepimage.parameters())
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

        if isinstance(optimizers, Dict) and not isinstance(lr_schedulers, Dict):
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

        if isinstance(model, str):
            self.model = torch.load(model)
        else:
            self.model = model
        self.verbose = verbose
        self.seed = seed
        self.early_stop = False
        self.objective = objective
        self.method = _ObjectiveToMethod.get(objective)

        self.loss_fn = self._get_loss_fn(
            objective, class_weight, custom_loss_function, alpha, gamma
        )
        self._initialize(initializers)
        self.optimizer = self._get_optimizer(optimizers)
        self.lr_scheduler, self.cyclic_lr = self._get_lr_scheduler(lr_schedulers)
        self.transforms = self._get_transforms(transforms)
        self._set_callbacks_and_metrics(callbacks, metrics)

        self.model.to(device)

    @Alias("finetune", ["warmup"])  # noqa: C901
    @Alias("finetune_epochs", ["warmup_epochs"])
    @Alias("finetune_max_lr", ["warmup_max_lr"])
    @Alias("finetune_deeptabular_gradual", ["warmup_deeptabular_gradual"])
    @Alias("finetune_deeptabular_max_lr", ["warmup_deeptabular_max_lr"])
    @Alias("finetune_deeptabular_layers", ["warmup_deeptabular_layers"])
    @Alias("finetune_deeptext_gradual", ["warmup_deeptext_gradual"])
    @Alias("finetune_deeptext_max_lr", ["warmup_deeptext_max_lr"])
    @Alias("finetune_deeptext_layers", ["warmup_deeptext_layers"])
    @Alias("finetune_deepimage_gradual", ["warmup_deepimage_gradual"])
    @Alias("finetune_deepimage_max_lr", ["warmup_deepimage_max_lr"])
    @Alias("finetune_deepimage_layers", ["warmup_deepimage_layers"])
    @Alias("finetune_routine", ["warmup_routine"])
    def fit(  # noqa: C901
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
        patience: int = 10,
        finetune: bool = False,
        finetune_epochs: int = 5,
        finetune_max_lr: float = 0.01,
        finetune_deeptabular_gradual: bool = False,
        finetune_deeptabular_max_lr: float = 0.01,
        finetune_deeptabular_layers: Optional[List[nn.Module]] = None,
        finetune_deeptext_gradual: bool = False,
        finetune_deeptext_max_lr: float = 0.01,
        finetune_deeptext_layers: Optional[List[nn.Module]] = None,
        finetune_deepimage_gradual: bool = False,
        finetune_deepimage_max_lr: float = 0.01,
        finetune_deepimage_layers: Optional[List[nn.Module]] = None,
        finetune_routine: str = "howard",
        stop_after_finetuning: bool = False,
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
        patience: int, default=10
            Number of epochs without improving the target metric or loss
            before the fit process stops
        finetune: bool, default=False
            param alias: ``warmup``

            fine-tune individual model components.

            .. note:: This functionality can also be used to 'warm-up'
               individual components before the joined training starts, and hence
               its alias. See the Examples folder in the repo for more details

            ``pytorch_widedeep`` implements 3 fine-tune routines.

            - fine-tune all trainable layers at once. This routine is is
              inspired by the work of Howard & Sebastian Ruder 2018 in their
              `ULMfit paper <https://arxiv.org/abs/1801.06146>`_. Using a
              Slanted Triangular learing (see `Leslie N. Smith paper
              <https://arxiv.org/pdf/1506.01186.pdf>`_), the process is the
              following: `i`) the learning rate will gradually increase for
              10% of the training steps from max_lr/10 to max_lr. `ii`) It
              will then gradually decrease to max_lr/10 for the remaining 90%
              of the steps. The optimizer used in the process is ``AdamW``.

            and two gradual fine-tune routines, where only certain layers are
            trained at a time.

            - The so called `Felbo` gradual fine-tune rourine, based on the the
              Felbo et al., 2017 `DeepEmoji paper <https://arxiv.org/abs/1708.00524>`_.
            - The `Howard` routine based on the work of Howard & Sebastian Ruder 2018 in their
              `ULMfit paper <https://arxiv.org/abs/1801.06146>`_.

            For details on how these routines work, please see the Examples
            section in this documentation and the `Examples
            <https://github.com/jrzaurin/pytorch-widedeep/tree/master/examples>`_
            folder in the repo.
        finetune_epochs: int, default=4
            param alias: ``warmup_epochs``

            Number of fine-tune epochs for those model components that will
            *NOT* be gradually fine-tuned. Those components with gradual
            fine-tune follow their corresponding specific routine.
        finetune_max_lr: float, default=0.01
            param alias: ``warmup_max_lr``

            Maximum learning rate during the Triangular Learning rate cycle
            for those model componenst that will *NOT* be gradually fine-tuned
        finetune_deeptabular_gradual: bool, default=False
            param alias: ``warmup_deeptabular_gradual``

            Boolean indicating if the ``deeptabular`` component will be
            fine-tuned gradually
        finetune_deeptabular_max_lr: float, default=0.01
            param alias: ``warmup_deeptabular_max_lr``

            Maximum learning rate during the Triangular Learning rate cycle
            for the deeptabular component
        finetune_deeptabular_layers: List, Optional, default=None
            param alias: ``warmup_deeptabular_layers``

            List of :obj:`nn.Modules` that will be fine-tuned gradually.

            .. note:: These have to be in `fine-tune-order`: the layers or blocks
                close to the output neuron(s) first

        finetune_deeptext_gradual: bool, default=False
            param alias: ``warmup_deeptext_gradual``

            Boolean indicating if the ``deeptext`` component will be
            fine-tuned gradually
        finetune_deeptext_max_lr: float, default=0.01
            param alias: ``warmup_deeptext_max_lr``

            Maximum learning rate during the Triangular Learning rate cycle
            for the deeptext component
        finetune_deeptext_layers: List, Optional, default=None
            param alias: ``warmup_deeptext_layers``

            List of :obj:`nn.Modules` that will be fine-tuned gradually.

            .. note:: These have to be in `fine-tune-order`: the layers or blocks
                close to the output neuron(s) first

        finetune_deepimage_gradual: bool, default=False
            param alias: ``warmup_deepimage_gradual``

            Boolean indicating if the ``deepimage`` component will be
            fine-tuned gradually
        finetune_deepimage_max_lr: float, default=0.01
            param alias: ``warmup_deepimage_max_lr``

            Maximum learning rate during the Triangular Learning rate cycle
            for the ``deepimage`` component
        finetune_deepimage_layers: List, Optional, default=None
            param alias: ``warmup_deepimage_layers``

            List of :obj:`nn.Modules` that will be fine-tuned gradually.

            .. note:: These have to be in `fine-tune-order`: the layers or blocks
                close to the output neuron(s) first

        finetune_routine: str, default = "howard"
            param alias: ``warmup_routine``

            Warm up routine. On of "felbo" or "howard". See the examples
            section in this documentation and the corresponding repo for
            details on how to use fine-tune routines

        Examples
        --------

        For a series of comprehensive examples please, see the `Examples
        <https://github.com/jrzaurin/pytorch-widedeep/tree/master/examples>`_
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

        self.batch_size = batch_size
        train_set, eval_set = self._train_val_split(
            X_wide, X_tab, X_text, X_img, X_train, X_val, val_split, target
        )
        train_loader = DataLoader(
            dataset=train_set, batch_size=batch_size, num_workers=n_cpus
        )
        train_steps = len(train_loader)
        if eval_set is not None:
            eval_loader = DataLoader(
                dataset=eval_set,
                batch_size=batch_size,
                num_workers=n_cpus,
                shuffle=False,
            )
            eval_steps = len(eval_loader)

        if finetune:
            self._finetune(
                train_loader,
                finetune_epochs,
                finetune_max_lr,
                finetune_deeptabular_gradual,
                finetune_deeptabular_layers,
                finetune_deeptabular_max_lr,
                finetune_deeptext_gradual,
                finetune_deeptext_layers,
                finetune_deeptext_max_lr,
                finetune_deepimage_gradual,
                finetune_deepimage_layers,
                finetune_deepimage_max_lr,
                finetune_routine,
            )
            if stop_after_finetuning:
                print("Fine-tuning finished")
                return
            else:
                if self.verbose:
                    print(
                        "Fine-tuning of individual components completed. "
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
                for batch_idx, (data, targett) in zip(t, train_loader):
                    t.set_description("epoch %i" % (epoch + 1))
                    score, train_loss = self._training_step(data, targett, batch_idx)
                    if score is not None:
                        t.set_postfix(
                            metrics={k: np.round(v, 4) for k, v in score.items()},
                            loss=train_loss,
                        )
                    else:
                        t.set_postfix(loss=train_loss)
                    if self.lr_scheduler:
                        self._lr_scheduler_step(step_location="on_batch_end")
                    self.callback_container.on_batch_end(batch=batch_idx)
            epoch_logs["train_loss"] = train_loss
            if score is not None:
                for k, v in score.items():
                    log_k = "_".join(["train", k])
                    epoch_logs[log_k] = v
            if eval_set is not None and epoch % validation_freq == (
                validation_freq - 1
            ):
                self.valid_running_loss = 0.0
                with trange(eval_steps, disable=self.verbose != 1) as v:
                    for i, (data, targett) in zip(v, eval_loader):
                        v.set_description("valid")
                        score, val_loss = self._validation_step(data, targett, i)
                        if score is not None:
                            v.set_postfix(
                                metrics={k: np.round(v, 4) for k, v in score.items()},
                                loss=val_loss,
                            )
                        else:
                            v.set_postfix(loss=val_loss)
                epoch_logs["val_loss"] = val_loss
                if score is not None:
                    for k, v in score.items():
                        log_k = "_".join(["val", k])
                        epoch_logs[log_k] = v
            if self.lr_scheduler:
                self._lr_scheduler_step(step_location="on_epoch_end")
            self.callback_container.on_epoch_end(epoch, epoch_logs)
            if self.early_stop:
                self.callback_container.on_train_end(epoch_logs)
                break
        self.callback_container.on_train_end(epoch_logs)
        self.model.train()

    def predict(  # type: ignore[return]
        self,
        X_wide: Optional[np.ndarray] = None,
        X_tab: Optional[np.ndarray] = None,
        X_text: Optional[np.ndarray] = None,
        X_img: Optional[np.ndarray] = None,
        X_test: Optional[Dict[str, np.ndarray]] = None,
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
        """

        preds_l = self._predict(X_wide, X_tab, X_text, X_img, X_test)
        if self.method == "regression":
            return np.vstack(preds_l).squeeze(1)
        if self.method == "binary":
            preds = np.vstack(preds_l).squeeze(1)
            return (preds > 0.5).astype("int")
        if self.method == "multiclass":
            preds = np.vstack(preds_l)
            return np.argmax(preds, 1)  # type: ignore[return-value]

    def predict_proba(  # type: ignore[return]
        self,
        X_wide: Optional[np.ndarray] = None,
        X_tab: Optional[np.ndarray] = None,
        X_text: Optional[np.ndarray] = None,
        X_img: Optional[np.ndarray] = None,
        X_test: Optional[Dict[str, np.ndarray]] = None,
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
        """

        preds_l = self._predict(X_wide, X_tab, X_text, X_img, X_test)
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
        <https://github.com/jrzaurin/pytorch-widedeep/tree/master/examples>`_
        folder in the repo

        For completion, here we include a `"fabricated"` example, i.e.
        assuming we have already trained the model, that we have the
        categorical encodings in a dictionary name ``encoding_dict``, and that
        there is a column called `'education'`:

        .. code-block:: python

            trainer.get_embeddings(col_name="education", cat_encoding_dict=encoding_dict)
        """
        for n, p in self.model.named_parameters():
            if "embed_layers" in n and col_name in n:
                embed_mtx = p.cpu().data.numpy()
        encoding_dict = cat_encoding_dict[col_name]
        inv_encoding_dict = {v: k for k, v in encoding_dict.items()}
        cat_embed_dict = {}
        for idx, value in inv_encoding_dict.items():
            cat_embed_dict[value] = embed_mtx[idx]
        return cat_embed_dict

    def save_model(self, path: str):
        """Saves the model to disk

        Parameters
        ----------
        path: str
            full path to the directory where the model will be saved
        """
        self._makedir_if_not_exist(path)
        torch.save(self.model, path)

    @staticmethod
    def load_model(path: str) -> nn.Module:
        """Loads the model from disk

        Parameters
        ----------
        path: str
            full path to the directory from where the model will be read
        """
        return torch.load(path)

    def save_model_state_dict(self, path: str):
        """Saves the state dictionary to disk

        Parameters
        ----------
        path: str
            full path to the directory where the model's state dictionary will
            be saved
        """
        self._makedir_if_not_exist(path)
        torch.save(self.model.state_dict(), path)

    def load_model_state_dict(self, path: str):
        """Saves the state dictionary to disk

        Parameters
        ----------
        path: str
            full path to the directory from where the model's state dictionary
            will be loaded
        """
        self.model.load_state_dict(torch.load(path))

    def _train_val_split(  # noqa: C901
        self,
        X_wide: Optional[np.ndarray] = None,
        X_tab: Optional[np.ndarray] = None,
        X_text: Optional[np.ndarray] = None,
        X_img: Optional[np.ndarray] = None,
        X_train: Optional[Dict[str, np.ndarray]] = None,
        X_val: Optional[Dict[str, np.ndarray]] = None,
        val_split: Optional[float] = None,
        target: Optional[np.ndarray] = None,
    ):
        r"""
        If a validation set (X_val) is passed to the fit method, or val_split
        is specified, the train/val split will happen internally. A number of
        options are allowed in terms of data inputs. For parameter
        information, please, see the .fit() method documentation

        Returns
        -------
        train_set: WideDeepDataset
            :obj:`WideDeepDataset` object that will be loaded through
            :obj:`torch.utils.data.DataLoader`. See
            :class:`pytorch_widedeep.models._wd_dataset`
        eval_set : WideDeepDataset
            :obj:`WideDeepDataset` object that will be loaded through
            :obj:`torch.utils.data.DataLoader`. See
            :class:`pytorch_widedeep.models._wd_dataset`
        """

        if X_val is not None:
            assert (
                X_train is not None
            ), "if the validation set is passed as a dictionary, the training set must also be a dictionary"
            train_set = WideDeepDataset(**X_train, transforms=self.transforms)  # type: ignore
            eval_set = WideDeepDataset(**X_val, transforms=self.transforms)  # type: ignore
        elif val_split is not None:
            if not X_train:
                X_train = self._build_train_dict(X_wide, X_tab, X_text, X_img, target)
            y_tr, y_val, idx_tr, idx_val = train_test_split(
                X_train["target"],
                np.arange(len(X_train["target"])),
                test_size=val_split,
                random_state=self.seed,
                stratify=X_train["target"] if self.method != "regression" else None,
            )
            X_tr, X_val = {"target": y_tr}, {"target": y_val}
            if "X_wide" in X_train.keys():
                X_tr["X_wide"], X_val["X_wide"] = (
                    X_train["X_wide"][idx_tr],
                    X_train["X_wide"][idx_val],
                )
            if "X_tab" in X_train.keys():
                X_tr["X_tab"], X_val["X_tab"] = (
                    X_train["X_tab"][idx_tr],
                    X_train["X_tab"][idx_val],
                )
            if "X_text" in X_train.keys():
                X_tr["X_text"], X_val["X_text"] = (
                    X_train["X_text"][idx_tr],
                    X_train["X_text"][idx_val],
                )
            if "X_img" in X_train.keys():
                X_tr["X_img"], X_val["X_img"] = (
                    X_train["X_img"][idx_tr],
                    X_train["X_img"][idx_val],
                )
            train_set = WideDeepDataset(**X_tr, transforms=self.transforms)  # type: ignore
            eval_set = WideDeepDataset(**X_val, transforms=self.transforms)  # type: ignore
        else:
            if not X_train:
                X_train = self._build_train_dict(X_wide, X_tab, X_text, X_img, target)
            train_set = WideDeepDataset(**X_train, transforms=self.transforms)  # type: ignore
            eval_set = None

        return train_set, eval_set

    @staticmethod
    def _build_train_dict(X_wide, X_tab, X_text, X_img, target):
        X_train = {"target": target}
        if X_wide is not None:
            X_train["X_wide"] = X_wide
        if X_tab is not None:
            X_train["X_tab"] = X_tab
        if X_text is not None:
            X_train["X_text"] = X_text
        if X_img is not None:
            X_train["X_img"] = X_img
        return X_train

    def _finetune(
        self,
        loader: DataLoader,
        n_epochs: int,
        max_lr: float,
        deeptabular_gradual: bool,
        deeptabular_layers: List[nn.Module],
        deeptabular_max_lr: float,
        deeptext_gradual: bool,
        deeptext_layers: List[nn.Module],
        deeptext_max_lr: float,
        deepimage_gradual: bool,
        deepimage_layers: List[nn.Module],
        deepimage_max_lr: float,
        routine: str = "felbo",
    ):  # pragma: no cover
        r"""
        Simple wrap-up to individually fine-tune model components
        """
        if self.model.deephead is not None:
            raise ValueError(
                "Currently warming up is only supported without a fully connected 'DeepHead'"
            )
        # This is not the most elegant solution, but is a soluton "in-between"
        # a non elegant one and re-factoring the whole code
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

    def _lr_scheduler_step(self, step_location: str):  # noqa: C901
        r"""
        Function to execute the learning rate schedulers steps.
        If the lr_scheduler is Cyclic (i.e. CyclicLR or OneCycleLR), the step
        must happen after training each bach durig training. On the other
        hand, if the  scheduler is not Cyclic, is expected to be called after
        validation. (Consider coding this function as callback)

        Parameters
        ----------
        step_location: Str
            Indicates where to run the lr_scheduler step
        """
        if (
            self.lr_scheduler.__class__.__name__ == "MultipleLRScheduler"
            and self.cyclic_lr
        ):
            if step_location == "on_batch_end":
                for model_name, scheduler in self.lr_scheduler._schedulers.items():  # type: ignore
                    if "cycl" in scheduler.__class__.__name__.lower():
                        scheduler.step()  # type: ignore
            elif step_location == "on_epoch_end":
                for scheduler_name, scheduler in self.lr_scheduler._schedulers.items():  # type: ignore
                    if "cycl" not in scheduler.__class__.__name__.lower():
                        scheduler.step()  # type: ignore
        elif self.cyclic_lr:
            if step_location == "on_batch_end":
                self.lr_scheduler.step()  # type: ignore
            else:
                pass
        elif self.lr_scheduler.__class__.__name__ == "MultipleLRScheduler":
            if step_location == "on_epoch_end":
                self.lr_scheduler.step()  # type: ignore
            else:
                pass
        elif step_location == "on_epoch_end":
            self.lr_scheduler.step()  # type: ignore
        else:
            pass

    def _training_step(self, data: Dict[str, Tensor], target: Tensor, batch_idx: int):
        self.model.train()
        X = {k: v.cuda() for k, v in data.items()} if use_cuda else data
        y = target.view(-1, 1).float() if self.method != "multiclass" else target
        y = y.to(device)

        self.optimizer.zero_grad()
        y_pred = self.model(X)
        loss = self.loss_fn(y_pred, y)
        loss.backward()
        self.optimizer.step()

        self.train_running_loss += loss.item()
        avg_loss = self.train_running_loss / (batch_idx + 1)

        return self._get_score(y_pred, y), avg_loss

    def _validation_step(self, data: Dict[str, Tensor], target: Tensor, batch_idx: int):

        self.model.eval()
        with torch.no_grad():
            X = {k: v.cuda() for k, v in data.items()} if use_cuda else data
            y = target.view(-1, 1).float() if self.method != "multiclass" else target
            y = y.to(device)

            y_pred = self.model(X)
            loss = self.loss_fn(y_pred, y)
            self.valid_running_loss += loss.item()
            avg_loss = self.valid_running_loss / (batch_idx + 1)

        return self._get_score(y_pred, y), avg_loss

    def _get_score(self, y_pred, y):
        if self.metric is not None:
            if self.method == "regression":
                score = self.metric(y_pred, y)
            if self.method == "binary":
                score = self.metric(torch.sigmoid(y_pred), y)
            if self.method == "multiclass":
                score = self.metric(F.softmax(y_pred, dim=1), y)
            return score
        else:
            return None

    def _predict(
        self,
        X_wide: Optional[np.ndarray] = None,
        X_tab: Optional[np.ndarray] = None,
        X_text: Optional[np.ndarray] = None,
        X_img: Optional[np.ndarray] = None,
        X_test: Optional[Dict[str, np.ndarray]] = None,
    ) -> List:
        r"""Private method to avoid code repetition in predict and
        predict_proba. For parameter information, please, see the .predict()
        method documentation
        """
        if X_test is not None:
            test_set = WideDeepDataset(**X_test)
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
            test_set = WideDeepDataset(**load_dict)

        test_loader = DataLoader(
            dataset=test_set,
            batch_size=self.batch_size,
            num_workers=n_cpus,
            shuffle=False,
        )
        test_steps = (len(test_loader.dataset) // test_loader.batch_size) + 1  # type: ignore[arg-type]

        self.model.eval()
        preds_l = []
        with torch.no_grad():
            with trange(test_steps, disable=self.verbose != 1) as t:
                for i, data in zip(t, test_loader):
                    t.set_description("predict")
                    X = {k: v.cuda() for k, v in data.items()} if use_cuda else data
                    preds = self.model(X)
                    if self.method == "binary":
                        preds = torch.sigmoid(preds)
                    if self.method == "multiclass":
                        preds = F.softmax(preds, dim=1)
                    preds = preds.cpu().data.numpy()
                    preds_l.append(preds)
        self.model.train()
        return preds_l

    @staticmethod
    def _makedir_if_not_exist(path):
        if len(path.split("/")[:-1]) == 0:
            raise ValueError(
                "'path' must be the full path to save the model, including"
                " the root of the filenames. e.g. 'model/model.t'"
            )
        root_dir = ("/").join(path.split("/")[:-1])
        if not os.path.exists(root_dir):
            os.makedirs(root_dir)

    @staticmethod
    def _alias_to_loss(loss_fn: str, **kwargs):
        if loss_fn not in _ObjectiveToMethod.keys():
            raise ValueError(
                "objective or loss function is not supported. Please consider passing a callable "
                "directly to the compile method (see docs) or use one of the supported objectives "
                "or loss functions: {}".format(", ".join(_ObjectiveToMethod.keys()))
            )
        if loss_fn in _LossAliases.get("binary"):
            return nn.BCEWithLogitsLoss(weight=kwargs["weight"])
        if loss_fn in _LossAliases.get("multiclass"):
            return nn.CrossEntropyLoss(weight=kwargs["weight"])
        if loss_fn in _LossAliases.get("regression"):
            return nn.MSELoss()
        if loss_fn in _LossAliases.get("mean_absolute_error"):
            return nn.L1Loss()
        if loss_fn in _LossAliases.get("mean_squared_log_error"):
            return MSLELoss()
        if loss_fn in _LossAliases.get("root_mean_squared_error"):
            return RMSELoss()
        if loss_fn in _LossAliases.get("root_mean_squared_log_error"):
            return RMSLELoss()
        if "focal_loss" in loss_fn:
            return FocalLoss(**kwargs)

    def _get_loss_fn(self, objective, class_weight, custom_loss_function, alpha, gamma):
        if isinstance(class_weight, float):
            class_weight = torch.tensor([1.0 - class_weight, class_weight])
        elif isinstance(class_weight, (tuple, list)):
            class_weight = torch.tensor(class_weight)
        else:
            class_weight = None
        if custom_loss_function is not None:
            return custom_loss_function
        elif self.method != "regression" and "focal_loss" not in objective:
            return self._alias_to_loss(objective, weight=class_weight)
        elif "focal_loss" in objective:
            return self._alias_to_loss(objective, alpha=alpha, gamma=gamma)
        else:
            return self._alias_to_loss(objective)

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

    def _get_optimizer(self, optimizers):
        if optimizers is not None:
            if isinstance(optimizers, Optimizer):
                optimizer: Union[Optimizer, MultipleOptimizer] = optimizers
            elif isinstance(optimizers, Dict):
                opt_names = list(optimizers.keys())
                mod_names = [n for n, c in self.model.named_children()]
                for mn in mod_names:
                    assert mn in opt_names, "No optimizer found for {}".format(mn)
                optimizer = MultipleOptimizer(optimizers)
        else:
            optimizer = torch.optim.AdamW(self.model.parameters())  # type: ignore
        return optimizer

    @staticmethod
    def _get_lr_scheduler(lr_schedulers):
        if lr_schedulers is not None:
            if isinstance(lr_schedulers, LRScheduler):
                lr_scheduler: Union[
                    LRScheduler,
                    MultipleLRScheduler,
                ] = lr_schedulers
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
        return lr_scheduler, cyclic_lr

    @staticmethod
    def _get_transforms(transforms):
        if transforms is not None:
            return MultipleTransforms(transforms)()
        else:
            return None

    def _set_callbacks_and_metrics(self, callbacks, metrics):
        self.callbacks: List = [History()]
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
