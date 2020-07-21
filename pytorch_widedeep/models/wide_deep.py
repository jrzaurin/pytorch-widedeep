import os
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import trange
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from ..losses import FocalLoss
from ._warmup import WarmUp
from ..metrics import Metric, MetricCallback, MultipleMetrics
from ..wdtypes import *
from ..callbacks import History, Callback, CallbackContainer
from .deep_dense import dense_layer
from ._wd_dataset import WideDeepDataset
from ..initializers import Initializer, MultipleInitializer
from ._multiple_optimizer import MultipleOptimizer
from ._multiple_transforms import MultipleTransforms
from ._multiple_lr_scheduler import MultipleLRScheduler

n_cpus = os.cpu_count()
use_cuda = torch.cuda.is_available()


class WideDeep(nn.Module):
    r"""Main collector class that combines all ``Wide``, ``DeepDense``,
    ``DeepText`` and ``DeepImage`` models.

    There are two options to combine these models that correspond to the two
    architectures that ``pytorch-widedeep`` can build.

        - Directly connecting the output of the model components to an ouput neuron(s).

        - Adding a `Fully-Connected Head` (FC-Head) on top of the deep models.
          This FC-Head will combine the output form the ``DeepDense``, ``DeepText`` and
          ``DeepImage`` and will be then connected to the output neuron(s).

    Parameters
    ----------
    wide: nn.Module
        Wide model. We recommend using the ``Wide`` class in this package.
        However, it is possible to use a custom model as long as is consistent
        with the required architecture, see
        :class:`pytorch_widedeep.models.wide.Wide`
    deepdense: nn.Module
        `Deep dense` model comprised by the embeddings for the categorical
        features combined with numerical (also referred as continuous)
        features. We recommend using the ``DeepDense`` class in this package.
        However, a custom model as long as is  consistent with the required
        architecture. See :class:`pytorch_widedeep.models.deep_dense.DeepDense`.
    deeptext: nn.Module, Optional
        `Deep text` model for the text input. Must be an object of class
        ``DeepText`` or a custom model as long as is consistent with the
        required architecture. See
        :class:`pytorch_widedeep.models.deep_dense.DeepText`
    deepimage: nn.Module, Optional
        `Deep Image` model for the images input. Must be an object of class
        ``DeepImage`` or a custom model as long as is consistent with the
        required architecture. See
        :class:`pytorch_widedeep.models.deep_dense.DeepImage`
    deephead: nn.Module, Optional
        `Dense` model consisting in a stack of dense layers. The FC-Head.
    head_layers: List, Optional
        Alternatively, we can use ``head_layers`` to specify the sizes of the
        stacked dense layers in the fc-head e.g: ``[128, 64]``
    head_dropout: List, Optional
        Dropout between the layers in ``head_layers``. e.g: ``[0.5, 0.5]``
    head_batchnorm: bool, Optional
        Specifies if batch normalizatin should be included in the dense layers
    pred_dim: int
        Size of the final wide and deep output layer containing the
        predictions. `1` for regression and binary classification or `number
        of classes` for multiclass classification.


    .. note:: With the exception of ``cyclic``, all attributes are direct assignations of
        the corresponding parameters used when calling ``compile``.  Therefore,
        see the parameters at
        :class:`pytorch_widedeep.models.wide_deep.WideDeep.compile` for a full
        list of the attributes of an instance of
        :class:`pytorch_widedeep.models.wide_deep.Wide`


    Attributes
    ----------
    cyclic: :obj:`bool`
        Attribute that indicates if any of the lr_schedulers is cyclic (i.e. ``CyclicLR`` or
        ``OneCycleLR``). See `Pytorch schedulers <https://pytorch.org/docs/stable/optim.html>`_.


    .. note:: While I recommend using the ``Wide`` and ``DeepDense`` classes within
        this package when building the corresponding model components, it is very
        likely that the user will want to use custom text and image models. That
        is perfectly possible. Simply, build them and pass them as the
        corresponding parameters. Note that the custom models MUST return a last
        layer of activations (i.e. not the final prediction) so that  these
        activations are collected by ``WideDeep`` and combined accordingly. In
        addition, the models MUST also contain an attribute ``output_dim`` with
        the size of these last layers of activations. See for example
        :class:`pytorch_widedeep.models.deep_dense.DeepDense`

    """

    def __init__(
        self,
        wide: nn.Module,
        deepdense: nn.Module,
        pred_dim: int = 1,
        deeptext: Optional[nn.Module] = None,
        deepimage: Optional[nn.Module] = None,
        deephead: Optional[nn.Module] = None,
        head_layers: Optional[List[int]] = None,
        head_dropout: Optional[List] = None,
        head_batchnorm: Optional[bool] = None,
    ):

        super(WideDeep, self).__init__()

        # check that model components have the required output_dim attribute
        if not hasattr(deepdense, "output_dim"):
            raise AttributeError(
                "deepdense model must have an 'output_dim' attribute. "
                "See pytorch-widedeep.models.deep_dense.DeepDense"
            )
        if deeptext is not None and not hasattr(deeptext, "output_dim"):
            raise AttributeError(
                "deeptext model must have an 'output_dim' attribute. "
                "See pytorch-widedeep.models.deep_dense.DeepText"
            )
        if deepimage is not None and not hasattr(deepimage, "output_dim"):
            raise AttributeError(
                "deepimage model must have an 'output_dim' attribute. "
                "See pytorch-widedeep.models.deep_dense.DeepText"
            )

        # The main 5 components of the wide and deep assemble
        self.wide = wide
        self.deepdense = deepdense
        self.deeptext = deeptext
        self.deepimage = deepimage
        self.deephead = deephead

        if deephead is not None and head_layers is not None:
            warnings.warn(
                "both 'deephead' and 'head_layers' are not None."
                "'deephead' takes priority and will be used",
                UserWarning,
            )

        if self.deephead is None:
            if head_layers is not None:
                input_dim: int = self.deepdense.output_dim  # type:ignore
                if self.deeptext is not None:
                    input_dim += self.deeptext.output_dim  # type:ignore
                if self.deepimage is not None:
                    input_dim += self.deepimage.output_dim  # type:ignore
                head_layers = [input_dim] + head_layers
                if not head_dropout:
                    head_dropout = [0.0] * (len(head_layers) - 1)
                self.deephead = nn.Sequential()
                for i in range(1, len(head_layers)):
                    self.deephead.add_module(
                        "head_layer_{}".format(i - 1),
                        dense_layer(
                            head_layers[i - 1],
                            head_layers[i],
                            head_dropout[i - 1],
                            head_batchnorm,
                        ),
                    )
                self.deephead.add_module(
                    "head_out", nn.Linear(head_layers[-1], pred_dim)
                )
            else:
                self.deepdense = nn.Sequential(
                    self.deepdense, nn.Linear(self.deepdense.output_dim, pred_dim)  # type: ignore
                )
                if self.deeptext is not None:
                    self.deeptext = nn.Sequential(
                        self.deeptext, nn.Linear(self.deeptext.output_dim, pred_dim)  # type: ignore
                    )
                if self.deepimage is not None:
                    self.deepimage = nn.Sequential(
                        self.deepimage, nn.Linear(self.deepimage.output_dim, pred_dim)  # type: ignore
                    )

    def forward(self, X: Dict[str, Tensor]) -> Tensor:  # type: ignore

        # Wide output: direct connection to the output neuron(s)
        out = self.wide(X["wide"])

        # Deep output: either connected directly to the output neuron(s) or
        # passed through a head first
        if self.deephead:
            deepside = self.deepdense(X["deepdense"])
            if self.deeptext is not None:
                deepside = torch.cat([deepside, self.deeptext(X["deeptext"])], axis=1)  # type: ignore
            if self.deepimage is not None:
                deepside = torch.cat([deepside, self.deepimage(X["deepimage"])], axis=1)  # type: ignore
            deepside_out = self.deephead(deepside)
            return out.add(deepside_out)
        else:
            out.add(self.deepdense(X["deepdense"]))
            if self.deeptext is not None:
                out.add(self.deeptext(X["deeptext"]))
            if self.deepimage is not None:
                out.add(self.deepimage(X["deepimage"]))
            return out

    def compile(
        self,
        method: str,
        optimizers: Optional[Union[Optimizer, Dict[str, Optimizer]]] = None,
        lr_schedulers: Optional[Union[LRScheduler, Dict[str, LRScheduler]]] = None,
        initializers: Optional[Dict[str, Initializer]] = None,
        transforms: Optional[List[Transforms]] = None,
        callbacks: Optional[List[Callback]] = None,
        metrics: Optional[List[Metric]] = None,
        class_weight: Optional[Union[float, List[float], Tuple[float]]] = None,
        with_focal_loss: bool = False,
        alpha: float = 0.25,
        gamma: float = 2,
        verbose: int = 1,
        seed: int = 1,
    ):
        r"""Method to set the of attributes that will be used during the
        training process.

        Parameters
        ----------
        method: str
            One of `regression`, `binary` or `multiclass`
        optimizers: Union[Optimizer, Dict[str, Optimizer]], Optional, Default=AdamW
            - An instance of ``pytorch``'s ``Optimizer`` object (e.g. :obj:`torch.optim.Adam()`) or
            - a dictionary where there keys are the model components (i.e.
              `'wide'`, `'deepdense'`, `'deeptext'`, `'deepimage'` and/or `'deephead'`)  and
              the values are the corresponding optimizers. If multiple optimizers are used
              the  dictionary MUST contain an optimizer per model component.

            See `Pytorch optimizers <https://pytorch.org/docs/stable/optim.html>`_.
        lr_schedulers: Union[LRScheduler, Dict[str, LRScheduler]], Optional, Default=None
            - An instance of ``pytorch``'s ``LRScheduler`` object (e.g
              :obj:`torch.optim.lr_scheduler.StepLR(opt, step_size=5)`) or
            - a dictionary where there keys are the model componenst (i.e. `'wide'`,
              `'deepdense'`, `'deeptext'`, `'deepimage'` and/or `'deephead'`) and the
              values are the corresponding learning rate schedulers.

            See `Pytorch schedulers <https://pytorch.org/docs/stable/optim.html>`_.
        initializers: Dict[str, Initializer], Optional. Default=None
            Dict where there keys are the model components (i.e. `'wide'`,
            `'deepdense'`, `'deeptext'`, `'deepimage'` and/or `'deephead'`) and the
            values are the corresponding initializers.
            See `Pytorch initializers <https://pytorch.org/docs/stable/nn.init.html>`_.
        transforms: List[Transforms], Optional. Default=None
            ``Transforms`` is a custom type. See
            :obj:`pytorch_widedeep.wdtypes`. List with
            :obj:`torchvision.transforms` to be applied to the image component
            of the model (i.e. ``deepimage``) See `torchvision transforms
            <https://pytorch.org/docs/stable/torchvision/transforms.html>`_.
        callbacks: List[Callback], Optional. Default=None
            Callbacks available are: ``ModelCheckpoint``, ``EarlyStopping``,
            and ``LRHistory``. The ``History`` callback is used by default.
            See the ``Callbacks`` section in this documentation or
            :obj:`pytorch_widedeep.callbacks`
        metrics: List[Metric], Optional. Default=None
            Metrics available are: ``Accuracy``, ``Precision``, ``Recall``,
            ``FBetaScore`` and ``F1Score``.  See the ``Metrics`` section in
            this documentation or :obj:`pytorch_widedeep.metrics`
        class_weight: Union[float, List[float], Tuple[float]]. Optional. Default=None
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
        with_focal_loss: bool, Optional. Default=False
            Boolean indicating whether to use the Focal Loss for highly imbalanced problems.
            For details on the focal loss see the `original paper
            <https://arxiv.org/pdf/1708.02002.pdf>`_.
        alpha: float. Default=0.25
            Focal Loss alpha parameter.
        gamma: float. Default=2
            Focal Loss gamma parameter.
        verbose: int
            Setting it to 0 will print nothing during training.
        seed: int, Default=1
            Random seed to be used throughout all the methods

        Example
        --------
        Assuming you have already built the model components (wide, deepdense, etc...)

        >>> from pytorch_widedeep.models import WideDeep
        >>> from pytorch_widedeep.initializers import *
        >>> from pytorch_widedeep.callbacks import *
        >>> from pytorch_widedeep.optim import RAdam
        >>> model = WideDeep(wide=wide, deepdense=deepdense, deeptext=deeptext, deepimage=deepimage)
        >>> wide_opt = torch.optim.Adam(model.wide.parameters())
        >>> deep_opt = torch.optim.Adam(model.deepdense.parameters())
        >>> text_opt = RAdam(model.deeptext.parameters())
        >>> img_opt  = RAdam(model.deepimage.parameters())
        >>> wide_sch = torch.optim.lr_scheduler.StepLR(wide_opt, step_size=5)
        >>> deep_sch = torch.optim.lr_scheduler.StepLR(deep_opt, step_size=3)
        >>> text_sch = torch.optim.lr_scheduler.StepLR(text_opt, step_size=5)
        >>> img_sch  = torch.optim.lr_scheduler.StepLR(img_opt, step_size=3)
        >>> optimizers = {'wide': wide_opt, 'deepdense':deep_opt, 'deeptext':text_opt, 'deepimage': img_opt}
        >>> schedulers = {'wide': wide_sch, 'deepdense':deep_sch, 'deeptext':text_sch, 'deepimage': img_sch}
        >>> initializers = {'wide': Uniform, 'deepdense':Normal, 'deeptext':KaimingNormal,
        >>> ... 'deepimage':KaimingUniform}
        >>> transforms = [ToTensor, Normalize(mean=mean, std=std)]
        >>> callbacks = [LRHistory, EarlyStopping, ModelCheckpoint(filepath='model_weights/wd_out.pt')]
        >>> model.compile(method='regression', initializers=initializers, optimizers=optimizers,
        >>> ... lr_schedulers=schedulers, callbacks=callbacks, transforms=transforms)
        """
        self.verbose = verbose
        self.seed = seed
        self.early_stop = False
        self.method = method
        self.with_focal_loss = with_focal_loss
        if self.with_focal_loss:
            self.alpha, self.gamma = alpha, gamma

        if isinstance(class_weight, float):
            self.class_weight = torch.tensor([1.0 - class_weight, class_weight])
        elif isinstance(class_weight, (tuple, list)):
            self.class_weight = torch.tensor(class_weight)
        else:
            self.class_weight = None

        if initializers is not None:
            self.initializer = MultipleInitializer(initializers, verbose=self.verbose)
            self.initializer.apply(self)

        if optimizers is not None:
            if isinstance(optimizers, Optimizer):
                self.optimizer: Union[Optimizer, MultipleOptimizer] = optimizers
            elif len(optimizers) > 1:
                opt_names = list(optimizers.keys())
                mod_names = [n for n, c in self.named_children()]
                for mn in mod_names:
                    assert mn in opt_names, "No optimizer found for {}".format(mn)
                self.optimizer = MultipleOptimizer(optimizers)
        else:
            self.optimizer = torch.optim.AdamW(self.parameters())  # type: ignore

        if lr_schedulers is not None:
            if isinstance(lr_schedulers, LRScheduler):
                self.lr_scheduler: Union[
                    LRScheduler, MultipleLRScheduler
                ] = lr_schedulers
                self.cyclic = "cycl" in self.lr_scheduler.__class__.__name__.lower()
            elif len(lr_schedulers) > 1:
                self.lr_scheduler = MultipleLRScheduler(lr_schedulers)
                scheduler_names = [
                    sc.__class__.__name__.lower()
                    for _, sc in self.lr_scheduler._schedulers.items()
                ]
                self.cyclic = any(["cycl" in sn for sn in scheduler_names])
        else:
            self.lr_scheduler, self.cyclic = None, False

        if transforms is not None:
            self.transforms: MultipleTransforms = MultipleTransforms(transforms)()
        else:
            self.transforms = None

        self.history = History()
        self.callbacks: List = [self.history]
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
        self.callback_container.set_model(self)

        if use_cuda:
            self.cuda()

    def fit(
        self,
        X_wide: Optional[np.ndarray] = None,
        X_deep: Optional[np.ndarray] = None,
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
        warm_up: bool = False,
        warm_epochs: int = 4,
        warm_max_lr: float = 0.01,
        warm_deeptext_gradual: bool = False,
        warm_deeptext_max_lr: float = 0.01,
        warm_deeptext_layers: Optional[List[nn.Module]] = None,
        warm_deepimage_gradual: bool = False,
        warm_deepimage_max_lr: float = 0.01,
        warm_deepimage_layers: Optional[List[nn.Module]] = None,
        warm_routine: str = "howard",
    ):
        r"""Fit method. Must run after calling ``compile``

        Parameters
        ----------
        X_wide: np.ndarray, Optional. Default=None
            Input for the ``wide`` model component.
            See :class:`pytorch_widedeep.preprocessing.WidePreprocessor`
        X_deep: np.ndarray, Optional. Default=None
            Input for the ``deepdense`` model component.
            See :class:`pytorch_widedeep.preprocessing.DensePreprocessor`
        X_text: np.ndarray, Optional. Default=None
            Input for the ``deeptext`` model component.
            See :class:`pytorch_widedeep.preprocessing.TextPreprocessor`
        X_img : np.ndarray, Optional. Default=None
            Input for the ``deepimage`` model component.
            See :class:`pytorch_widedeep.preprocessing.ImagePreprocessor`
        X_train: Dict[str, np.ndarray], Optional. Default=None
            Training dataset for the different model components. Keys are
            `X_wide`, `'X_deep'`, `'X_text'`, `'X_img'` and `'target'`. Values are
            the corresponding matrices.
        X_val: Dict, Optional. Default=None
            Validation dataset for the different model component. Keys are
            `'X_wide'`, `'X_deep'`, `'X_text'`, `'X_img'` and `'target'`. Values are
            the corresponding matrices.
        val_split: float, Optional. Default=None
            train/val split fraction
        target: np.ndarray, Optional. Default=None
            target values
        n_epochs: int, Default=1
            number of epochs
        validation_freq: int, Default=1
            epochs validation frequency
        batch_size: int, Default=32
        patience: int, Default=10
            Number of epochs without improving the target metric before
            the fit process stops
        warm_up: bool, Default=False
            warm up model components individually before the joined training
            starts.

            ``pytorch_widedeep`` implements 3 warm up routines.

            - Warm up all trainable layers at once. This routine is is
              inspired by the work of Howard & Sebastian Ruder 2018 in their
              `ULMfit paper <https://arxiv.org/abs/1801.06146>`_. Using a
              Slanted Triangular learing (see `Leslie N. Smith paper
              <https://arxiv.org/pdf/1506.01186.pdf>`_), the process is the
              following: `i`) the learning rate will gradually increase for
              10% of the training steps from max_lr/10 to max_lr. `ii`) It
              will then gradually decrease to max_lr/10 for the remaining 90%
              of the steps. The optimizer used in the process is ``AdamW``.

            and two gradual warm up routines, where only certain layers are
            warmed up at each warm up step.

            - The so called `Felbo` gradual warm up rourine, inpired by the Felbo et al., 2017
              `DeepEmoji paper <https://arxiv.org/abs/1708.00524>`_.
            - The `Howard` routine based on the work of Howard & Sebastian Ruder 2018 in their
              `ULMfit paper <https://arxiv.org/abs/1801.06146>`_.

            For details on how these routines work, please see the examples
            section in this documentation and the corresponding repo.
        warm_epochs: int, Default=4
            Number of warm up epochs for those model components that will NOT
            be gradually warmed up. Those components with gradual warm up
            follow their corresponding specific routine.
        warm_max_lr: float, Default=0.01
            Maximum learning rate during the Triangular Learning rate cycle
            for those model componenst that will NOT be gradually warmed up
        warm_deeptext_gradual: bool, Default=False
            Boolean indicating if the deeptext component will be warmed
            up gradually
        warm_deeptext_max_lr: float, Default=0.01
            Maximum learning rate during the Triangular Learning rate cycle
            for the deeptext component
        warm_deeptext_layers: List, Optional, Default=None
            List of :obj:`nn.Modules` that will be warmed up gradually.

            .. note:: These have to be in `warm-up-order`: the layers or blocks
                close to the output neuron(s) first

        warm_deepimage_gradual: bool, Default=False
            Boolean indicating if the deepimage component will be warmed
            up gradually
        warm_deepimage_max_lr: Float, Default=0.01
            Maximum learning rate during the Triangular Learning rate cycle
            for the deepimage component
        warm_deepimage_layers: List, Optional, Default=None
            List of :obj:`nn.Modules` that will be warmed up gradually.

            .. note:: These have to be in `warm-up-order`: the layers or blocks
                close to the output neuron(s) first

        warm_routine: str, Default=`felbo`
            Warm up routine. On of `felbo` or `howard`. See the examples
            section in this documentation and the corresponding repo for
            details on how to use warm up routines

        Examples
        --------
        Assuming you have already built and compiled the model


        >>> # Ex 1. using train input arrays directly and no validation
        >>> model.fit(X_wide=X_wide, X_deep=X_deep, target=target, n_epochs=10, batch_size=256)


        >>> # Ex 2: using train input arrays directly and validation with val_split
        >>> model.fit(X_wide=X_wide, X_deep=X_deep, target=target, n_epochs=10, batch_size=256, val_split=0.2)


        >>> # Ex 3: using train dict and val_split
        >>> X_train = {'X_wide': X_wide, 'X_deep': X_deep, 'target': y}
        >>> model.fit(X_train, n_epochs=10, batch_size=256, val_split=0.2)


        >>> # Ex 4: validation using training and validation dicts
        >>> X_train = {'X_wide': X_wide_tr, 'X_deep': X_deep_tr, 'target': y_tr}
        >>> X_val = {'X_wide': X_wide_val, 'X_deep': X_deep_val, 'target': y_val}
        >>> model.fit(X_train=X_train, X_val=X_val n_epochs=10, batch_size=256)

        .. note:: :obj:`WideDeep` assumes that `X_wide`, `X_deep` and `target` ALWAYS exist, while
            `X_text` and `X_img` are optional

        .. note:: Either `X_train` or the three `X_wide`, `X_deep` and `target` must be passed to the
            fit method

        """

        if X_train is None and (X_wide is None or X_deep is None or target is None):
            raise ValueError(
                "Training data is missing. Either a dictionary (X_train) with "
                "the training dataset or at least 3 arrays (X_wide, X_deep, "
                "target) must be passed to the fit method"
            )

        self.batch_size = batch_size
        train_set, eval_set = self._train_val_split(
            X_wide, X_deep, X_text, X_img, X_train, X_val, val_split, target
        )
        train_loader = DataLoader(
            dataset=train_set, batch_size=batch_size, num_workers=n_cpus
        )
        if warm_up:
            # warm up...
            self._warm_up(
                train_loader,
                warm_epochs,
                warm_max_lr,
                warm_deeptext_gradual,
                warm_deeptext_layers,
                warm_deeptext_max_lr,
                warm_deepimage_gradual,
                warm_deepimage_layers,
                warm_deepimage_max_lr,
                warm_routine,
            )
        train_steps = len(train_loader)
        self.callback_container.on_train_begin(
            {"batch_size": batch_size, "train_steps": train_steps, "n_epochs": n_epochs}
        )
        if self.verbose:
            print("Training")
        for epoch in range(n_epochs):
            # train step...
            epoch_logs: Dict[str, float] = {}
            self.callback_container.on_epoch_begin(epoch, logs=epoch_logs)
            self.train_running_loss = 0.0
            with trange(train_steps, disable=self.verbose != 1) as t:
                for batch_idx, (data, target) in zip(t, train_loader):
                    t.set_description("epoch %i" % (epoch + 1))
                    score, train_loss = self._training_step(data, target, batch_idx)
                    if score is not None:
                        t.set_postfix(
                            metrics={k: np.round(v, 4) for k, v in score.items()},
                            loss=train_loss,
                        )
                    else:
                        t.set_postfix(loss=np.sqrt(train_loss))
                    if self.lr_scheduler:
                        self._lr_scheduler_step(step_location="on_batch_end")
                    self.callback_container.on_batch_end(batch=batch_idx)
            epoch_logs["train_loss"] = train_loss
            if score is not None:
                for k, v in score.items():
                    log_k = "_".join(["train", k])
                    epoch_logs[log_k] = v
            # eval step...
            if epoch % validation_freq == (validation_freq - 1):
                if eval_set is not None:
                    eval_loader = DataLoader(
                        dataset=eval_set,
                        batch_size=batch_size,
                        num_workers=n_cpus,
                        shuffle=False,
                    )
                    eval_steps = len(eval_loader)
                    self.valid_running_loss = 0.0
                    with trange(eval_steps, disable=self.verbose != 1) as v:
                        for i, (data, target) in zip(v, eval_loader):
                            v.set_description("valid")
                            score, val_loss = self._validation_step(data, target, i)
                            if score is not None:
                                v.set_postfix(
                                    metrics={
                                        k: np.round(v, 4) for k, v in score.items()
                                    },
                                    loss=val_loss,
                                )
                            else:
                                v.set_postfix(loss=np.sqrt(val_loss))
                    epoch_logs["val_loss"] = val_loss
                    if score is not None:
                        for k, v in score.items():
                            log_k = "_".join(["val", k])
                            epoch_logs[log_k] = v
            if self.lr_scheduler:
                self._lr_scheduler_step(step_location="on_epoch_end")
            #  log and check if early_stop...
            self.callback_container.on_epoch_end(epoch, epoch_logs)
            if self.early_stop:
                self.callback_container.on_train_end(epoch_logs)
                break
            self.callback_container.on_train_end(epoch_logs)
        self.train()

    def predict(
        self,
        X_wide: np.ndarray,
        X_deep: np.ndarray,
        X_text: Optional[np.ndarray] = None,
        X_img: Optional[np.ndarray] = None,
        X_test: Optional[Dict[str, np.ndarray]] = None,
    ) -> np.ndarray:
        r"""Returns the predictions

        Parameters
        ----------
        X_wide: np.ndarray, Optional. Default=None
            Input for the ``wide`` model component.
            See :class:`pytorch_widedeep.preprocessing.WidePreprocessor`
        X_deep: np.ndarray, Optional. Default=None
            Input for the ``deepdense`` model component.
            See :class:`pytorch_widedeep.preprocessing.DensePreprocessor`
        X_text: np.ndarray, Optional. Default=None
            Input for the ``deeptext`` model component.
            See :class:`pytorch_widedeep.preprocessing.TextPreprocessor`
        X_img : np.ndarray, Optional. Default=None
            Input for the ``deepimage`` model component.
            See :class:`pytorch_widedeep.preprocessing.ImagePreprocessor`
        X_test: Dict[str, np.ndarray], Optional. Default=None
            Testing dataset for the different model components. Keys are
            `'X_wide'`, `'X_deep'`, `'X_text'`, `'X_img'` and `'target'` the values are
            the corresponding matrices.

        .. note:: WideDeep assumes that `X_wide`, `X_deep` and `target` ALWAYS exist,
            while `X_text` and `X_img` are optional.
        """
        preds_l = self._predict(X_wide, X_deep, X_text, X_img, X_test)
        if self.method == "regression":
            return np.vstack(preds_l).squeeze(1)
        if self.method == "binary":
            preds = np.vstack(preds_l).squeeze(1)
            return (preds > 0.5).astype("int")
        if self.method == "multiclass":
            preds = np.vstack(preds_l)
            return np.argmax(preds, 1)

    def predict_proba(
        self,
        X_wide: np.ndarray,
        X_deep: np.ndarray,
        X_text: Optional[np.ndarray] = None,
        X_img: Optional[np.ndarray] = None,
        X_test: Optional[Dict[str, np.ndarray]] = None,
    ) -> np.ndarray:
        r"""Returns the predicted probabilities for the test dataset for  binary
            and multiclass methods
        """
        preds_l = self._predict(X_wide, X_deep, X_text, X_img, X_test)
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
    ) -> Dict[str, np.ndarray]:
        r"""Returns the learned embeddings for the categorical features passed through
        ``deepdense``.

        This method is designed to take an encoding dictionary in the same
        format as that of the :obj:`LabelEncoder` Attribute of the class
        :obj:`DensePreprocessor`. See
        :class:`pytorch_widedeep.preprocessing.DensePreprocessor` and
        :class:`pytorch_widedeep.utils.dense_utils.LabelEncder`.

        Parameters
        ----------
        col_name: str,
            Column name of the feature we want to get the embeddings for
        cat_encoding_dict: Dict[str, Dict[str, int]]
            Dictionary containing the categorical encodings, e.g:

            Examples
            --------
            >>> cat_encoding_dict['education']
            {'11th': 0, 'HS-grad': 1, 'Assoc-acdm': 2, 'Some-college': 3, '10th': 4, 'Prof-school': 5,
            '7th-8th': 6, 'Bachelors': 7, 'Masters': 8, 'Doctorate': 9, '5th-6th': 10, 'Assoc-voc': 11,
            '9th': 12, '12th': 13, '1st-4th': 14, 'Preschool': 15}

        Examples
        --------

        Assuming we have already train the model and that we have the
        categorical encodings in a dictionary name ``encoding_dict``:

        >>> model.get_embeddings(col_name='education', cat_encoding_dict=encoding_dict)
        {'11th': array([-0.42739448, -0.22282735,  0.36969638,  0.4445322 ,  0.2562272 ,
        0.11572784, -0.01648579,  0.09027119,  0.0457597 , -0.28337458], dtype=float32),
         'HS-grad': array([-0.10600474, -0.48775527,  0.3444158 ,  0.13818645, -0.16547225,
        0.27409762, -0.05006042, -0.0668492 , -0.11047247,  0.3280354 ], dtype=float32),
        ...
        }
        """
        for n, p in self.named_parameters():
            if "embed_layers" in n and col_name in n:
                embed_mtx = p.cpu().data.numpy()
        encoding_dict = cat_encoding_dict[col_name]
        inv_encoding_dict = {v: k for k, v in encoding_dict.items()}
        cat_embed_dict = {}
        for idx, value in inv_encoding_dict.items():
            cat_embed_dict[value] = embed_mtx[idx]
        return cat_embed_dict

    def _loss_fn(self, y_pred: Tensor, y_true: Tensor) -> Tensor:  # type: ignore
        if self.with_focal_loss:
            return FocalLoss(self.alpha, self.gamma)(y_pred, y_true)
        if self.method == "regression":
            return F.mse_loss(y_pred, y_true.view(-1, 1))
        if self.method == "binary":
            return F.binary_cross_entropy_with_logits(
                y_pred, y_true.view(-1, 1), weight=self.class_weight
            )
        if self.method == "multiclass":
            return F.cross_entropy(y_pred, y_true, weight=self.class_weight)

    def _train_val_split(
        self,
        X_wide: Optional[np.ndarray] = None,
        X_deep: Optional[np.ndarray] = None,
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
        #  Without validation
        if X_val is None and val_split is None:
            # if a train dictionary is passed, check if text and image datasets
            # are present and instantiate the WideDeepDataset class
            if X_train is not None:
                X_wide, X_deep, target = (
                    X_train["X_wide"],
                    X_train["X_deep"],
                    X_train["target"],
                )
                if "X_text" in X_train.keys():
                    X_text = X_train["X_text"]
                if "X_img" in X_train.keys():
                    X_img = X_train["X_img"]
            X_train = {"X_wide": X_wide, "X_deep": X_deep, "target": target}
            try:
                X_train.update({"X_text": X_text})
            except:
                pass
            try:
                X_train.update({"X_img": X_img})
            except:
                pass
            train_set = WideDeepDataset(**X_train, transforms=self.transforms)  # type: ignore
            eval_set = None
        #  With validation
        else:
            if X_val is not None:
                # if a validation dictionary is passed, then if not train
                # dictionary is passed we build it with the input arrays
                # (either the dictionary or the arrays must be passed)
                if X_train is None:
                    X_train = {"X_wide": X_wide, "X_deep": X_deep, "target": target}
                    if X_text is not None:
                        X_train.update({"X_text": X_text})
                    if X_img is not None:
                        X_train.update({"X_img": X_img})
            else:
                # if a train dictionary is passed, check if text and image
                # datasets are present. The train/val split using val_split
                if X_train is not None:
                    X_wide, X_deep, target = (
                        X_train["X_wide"],
                        X_train["X_deep"],
                        X_train["target"],
                    )
                    if "X_text" in X_train.keys():
                        X_text = X_train["X_text"]
                    if "X_img" in X_train.keys():
                        X_img = X_train["X_img"]
                (
                    X_tr_wide,
                    X_val_wide,
                    X_tr_deep,
                    X_val_deep,
                    y_tr,
                    y_val,
                ) = train_test_split(
                    X_wide,
                    X_deep,
                    target,
                    test_size=val_split,
                    random_state=self.seed,
                    stratify=target if self.method != "regression" else None,
                )
                X_train = {"X_wide": X_tr_wide, "X_deep": X_tr_deep, "target": y_tr}
                X_val = {"X_wide": X_val_wide, "X_deep": X_val_deep, "target": y_val}
                try:
                    X_tr_text, X_val_text = train_test_split(
                        X_text,
                        test_size=val_split,
                        random_state=self.seed,
                        stratify=target if self.method != "regression" else None,
                    )
                    X_train.update({"X_text": X_tr_text}), X_val.update(
                        {"X_text": X_val_text}
                    )
                except:
                    pass
                try:
                    X_tr_img, X_val_img = train_test_split(
                        X_img,
                        test_size=val_split,
                        random_state=self.seed,
                        stratify=target if self.method != "regression" else None,
                    )
                    X_train.update({"X_img": X_tr_img}), X_val.update(
                        {"X_img": X_val_img}
                    )
                except:
                    pass
            # At this point the X_train and X_val dictionaries have been built
            train_set = WideDeepDataset(**X_train, transforms=self.transforms)  # type: ignore
            eval_set = WideDeepDataset(**X_val, transforms=self.transforms)  # type: ignore
        return train_set, eval_set

    def _warm_up(
        self,
        loader: DataLoader,
        n_epochs: int,
        max_lr: float,
        deeptext_gradual: bool,
        deeptext_layers: List[nn.Module],
        deeptext_max_lr: float,
        deepimage_gradual: bool,
        deepimage_layers: List[nn.Module],
        deepimage_max_lr: float,
        routine: str = "felbo",
    ):
        r"""
        Simple wrappup to individually warm up model components
        """
        if self.deephead is not None:
            raise ValueError(
                "Currently warming up is only supported without a fully connected 'DeepHead'"
            )
        # This is not the most elegant solution, but is a soluton "in-between"
        # a non elegant one and re-factoring the whole code
        warmer = WarmUp(self._loss_fn, self.metric, self.method, self.verbose)
        warmer.warm_all(self.wide, "wide", loader, n_epochs, max_lr)
        warmer.warm_all(self.deepdense, "deepdense", loader, n_epochs, max_lr)
        if self.deeptext:
            if deeptext_gradual:
                warmer.warm_gradual(
                    self.deeptext,
                    "deeptext",
                    loader,
                    deeptext_max_lr,
                    deeptext_layers,
                    routine,
                )
            else:
                warmer.warm_all(self.deeptext, "deeptext", loader, n_epochs, max_lr)
        if self.deepimage:
            if deepimage_gradual:
                warmer.warm_gradual(
                    self.deepimage,
                    "deepimage",
                    loader,
                    deepimage_max_lr,
                    deepimage_layers,
                    routine,
                )
            else:
                warmer.warm_all(self.deepimage, "deepimage", loader, n_epochs, max_lr)

    def _lr_scheduler_step(self, step_location: str):
        r"""
        Function to execute the learning rate schedulers steps.
        If the lr_scheduler is Cyclic (i.e. CyclicLR or OneCycleLR), the step
        must happen after training each bach durig training. On the other
        hand, if the  scheduler is not Cyclic, is expected to be called after
        validation.

        Parameters
        ----------
        step_location: Str
            Indicates where to run the lr_scheduler step
        """
        if (
            self.lr_scheduler.__class__.__name__ == "MultipleLRScheduler"
            and self.cyclic
        ):
            if step_location == "on_batch_end":
                for model_name, scheduler in self.lr_scheduler._schedulers.items():  # type: ignore
                    if "cycl" in scheduler.__class__.__name__.lower():
                        scheduler.step()  # type: ignore
            elif step_location == "on_epoch_end":
                for scheduler_name, scheduler in self.lr_scheduler._schedulers.items():  # type: ignore
                    if "cycl" not in scheduler.__class__.__name__.lower():
                        scheduler.step()  # type: ignore
        elif self.cyclic:
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
        self.train()
        X = {k: v.cuda() for k, v in data.items()} if use_cuda else data
        y = target.float() if self.method != "multiclass" else target
        y = y.cuda() if use_cuda else y

        self.optimizer.zero_grad()
        y_pred = self.forward(X)
        loss = self._loss_fn(y_pred, y)
        loss.backward()
        self.optimizer.step()

        self.train_running_loss += loss.item()
        avg_loss = self.train_running_loss / (batch_idx + 1)

        if self.metric is not None:
            if self.method == "binary":
                score = self.metric(torch.sigmoid(y_pred), y)
            if self.method == "multiclass":
                score = self.metric(F.softmax(y_pred, dim=1), y)
            return score, avg_loss
        else:
            return None, avg_loss

    def _validation_step(self, data: Dict[str, Tensor], target: Tensor, batch_idx: int):

        self.eval()
        with torch.no_grad():
            X = {k: v.cuda() for k, v in data.items()} if use_cuda else data
            y = target.float() if self.method != "multiclass" else target
            y = y.cuda() if use_cuda else y

            y_pred = self.forward(X)
            loss = self._loss_fn(y_pred, y)
            self.valid_running_loss += loss.item()
            avg_loss = self.valid_running_loss / (batch_idx + 1)

        if self.metric is not None:
            if self.method == "binary":
                score = self.metric(torch.sigmoid(y_pred), y)
            if self.method == "multiclass":
                score = self.metric(F.softmax(y_pred, dim=1), y)
            return score, avg_loss
        else:
            return None, avg_loss

    def _predict(
        self,
        X_wide: np.ndarray,
        X_deep: np.ndarray,
        X_text: Optional[np.ndarray] = None,
        X_img: Optional[np.ndarray] = None,
        X_test: Optional[Dict[str, np.ndarray]] = None,
    ) -> List:
        r"""Hidden method to avoid code repetition in predict and
        predict_proba. For parameter information, please, see the .predict()
        method documentation
        """
        if X_test is not None:
            test_set = WideDeepDataset(**X_test)
        else:
            load_dict = {"X_wide": X_wide, "X_deep": X_deep}
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
        test_steps = (len(test_loader.dataset) // test_loader.batch_size) + 1

        self.eval()
        preds_l = []
        with torch.no_grad():
            with trange(test_steps, disable=self.verbose != 1) as t:
                for i, data in zip(t, test_loader):
                    t.set_description("predict")
                    X = {k: v.cuda() for k, v in data.items()} if use_cuda else data
                    preds = self.forward(X)
                    if self.method == "binary":
                        preds = torch.sigmoid(preds)
                    if self.method == "multiclass":
                        preds = F.softmax(preds, dim=1)
                    preds = preds.cpu().data.numpy()
                    preds_l.append(preds)
        self.train()
        return preds_l
