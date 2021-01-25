import os
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import trange
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from ..losses import get_loss_function, objective_to_method
from ..models import WideDeep
from ._warmup import WarmUp
from ..metrics import Metric, MetricCallback, MultipleMetrics
from ..wdtypes import *  # noqa: F403
from ..callbacks import History, Callback, CallbackContainer
from ._wd_dataset import WideDeepDataset
from ..initializers import Initializer, MultipleInitializer
from ._multiple_optimizer import MultipleOptimizer
from ..utils.general_utils import Alias
from ._multiple_transforms import MultipleTransforms
from ._multiple_lr_scheduler import MultipleLRScheduler

warnings.filterwarnings("default", category=DeprecationWarning)

n_cpus = os.cpu_count()

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


class Trainer(object):
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
        initializers: Optional[Dict[str, Initializer]] = None,
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
        objective: str
            One of `regression`, `binary` or `multiclass`. The default when
            performing a `regression`, a `binary` classification or a
            `multiclass` classification is the `mean squared error
            <https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.mse_loss>`_
            (MSE), `Binary Cross Entropy
            <https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.binary_cross_entropy>`_
            (BCE) and `Cross Entropy
            <https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.cross_entropy>`_
            (CE) respectively.
        optimizers: Union[Optimizer, Dict[str, Optimizer]], Optional, Default=AdamW
            - An instance of ``pytorch``'s ``Optimizer`` object (e.g. :obj:`torch.optim.Adam()`) or
            - a dictionary where there keys are the model components (i.e.
              `'wide'`, `'deeptabular'`, `'deeptext'`, `'deepimage'` and/or `'deephead'`)  and
              the values are the corresponding optimizers. If multiple optimizers are used
              the  dictionary MUST contain an optimizer per model component.

            See `Pytorch optimizers <https://pytorch.org/docs/stable/optim.html>`_.
        lr_schedulers: Union[LRScheduler, Dict[str, LRScheduler]], Optional, Default=None
            - An instance of ``pytorch``'s ``LRScheduler`` object (e.g
              :obj:`torch.optim.lr_scheduler.StepLR(opt, step_size=5)`) or
            - a dictionary where there keys are the model componenst (i.e. `'wide'`,
              `'deeptabular'`, `'deeptext'`, `'deepimage'` and/or `'deephead'`) and the
              values are the corresponding learning rate schedulers.

            See `Pytorch schedulers <https://pytorch.org/docs/stable/optim.html>`_.
        initializers: Dict[str, Initializer], Optional. Default=None
            Dict where there keys are the model components (i.e. `'wide'`,
            `'deeptabular'`, `'deeptext'`, `'deepimage'` and/or `'deephead'`) and the
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
        >>> import torch
        >>> from torchvision.transforms import ToTensor
        >>>
        >>> from pytorch_widedeep.callbacks import EarlyStopping, LRHistory
        >>> from pytorch_widedeep.initializers import KaimingNormal, KaimingUniform, Normal, Uniform
        >>> from pytorch_widedeep.models import TabResnet, DeepImage, DeepText, Wide, WideDeep
        >>> from pytorch_widedeep.optim import RAdam
        >>> embed_input = [(u, i, j) for u, i, j in zip(["a", "b", "c"][:4], [4] * 3, [8] * 3)]
        >>> deep_column_idx = {k: v for v, k in enumerate(["a", "b", "c"])}
        >>> wide = Wide(10, 1)
        >>> deeptabular = TabResnet(blocks=[8, 4], deep_column_idx=deep_column_idx, embed_input=embed_input)
        >>> deeptext = DeepText(vocab_size=10, embed_dim=4, padding_idx=0)
        >>> deepimage = DeepImage(pretrained=False)
        >>> model = WideDeep(wide=wide, deeptabular=deeptabular, deeptext=deeptext, deepimage=deepimage)
        >>>
        >>> wide_opt = torch.optim.Adam(model.wide.parameters())
        >>> deep_opt = torch.optim.Adam(model.deeptabular.parameters())
        >>> text_opt = RAdam(model.deeptext.parameters())
        >>> img_opt = RAdam(model.deepimage.parameters())
        >>>
        >>> wide_sch = torch.optim.lr_scheduler.StepLR(wide_opt, step_size=5)
        >>> deep_sch = torch.optim.lr_scheduler.StepLR(deep_opt, step_size=3)
        >>> text_sch = torch.optim.lr_scheduler.StepLR(text_opt, step_size=5)
        >>> img_sch = torch.optim.lr_scheduler.StepLR(img_opt, step_size=3)
        >>> optimizers = {"wide": wide_opt, "deeptabular": deep_opt, "deeptext": text_opt, "deepimage": img_opt}
        >>> schedulers = {"wide": wide_sch, "deeptabular": deep_sch, "deeptext": text_sch, "deepimage": img_sch}
        >>> initializers = {"wide": Uniform, "deeptabular": Normal, "deeptext": KaimingNormal, "deepimage": KaimingUniform}
        >>> transforms = [ToTensor]
        >>> callbacks = [LRHistory(n_epochs=4), EarlyStopping]
        >>> model.compile(objective="regression", initializers=initializers, optimizers=optimizers,
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
                "If 'custom_loss_function' is not None, 'objective' might be 'binary' "
                "'multiclass' or 'regression', consistent with the loss function"
            )

        self.model = model
        self.verbose = verbose
        self.seed = seed
        self.early_stop = False
        self.objective = objective
        self.method = objective_to_method[objective]

        self.loss_fn = self._get_loss_fn(
            objective, class_weight, custom_loss_function, alpha, gamma
        )
        self._initialize(initializers)
        self.optimizer = self._get_optimizer(optimizers)
        self.lr_scheduler, self.cyclic_lr = self._get_lr_scheduler(lr_schedulers)
        self.transforms = self._get_transforms(transforms)
        self._set_callbacks_and_metrics(callbacks, metrics)

        self.model.to(device)

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
        X_tab: np.ndarray, Optional. Default=None
            Input for the ``deeptabular`` model component.
            See :class:`pytorch_widedeep.preprocessing.TabPreprocessor`
        X_text: np.ndarray, Optional. Default=None
            Input for the ``deeptext`` model component.
            See :class:`pytorch_widedeep.preprocessing.TextPreprocessor`
        X_img : np.ndarray, Optional. Default=None
            Input for the ``deepimage`` model component.
            See :class:`pytorch_widedeep.preprocessing.ImagePreprocessor`
        X_train: Dict[str, np.ndarray], Optional. Default=None
            Training dataset for the different model components. Keys are
            `X_wide`, `'X_tab'`, `'X_text'`, `'X_img'` and `'target'`. Values are
            the corresponding matrices.
        X_val: Dict, Optional. Default=None
            Validation dataset for the different model component. Keys are
            `'X_wide'`, `'X_tab'`, `'X_text'`, `'X_img'` and `'target'`. Values are
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

        For a series of comprehensive examples please, see the `example
        <https://github.com/jrzaurin/pytorch-widedeep/tree/master/examples>`_.
        folder in the repo

        For completion, here we include some `"fabricated"` examples, i.e. these assume
        you have already built and compiled the model


        >>> # Ex 1. using train input arrays directly and no validation
        >>> # model.fit(X_wide=X_wide, X_tab=X_tab, target=target, n_epochs=10, batch_size=256)


        >>> # Ex 2: using train input arrays directly and validation with val_split
        >>> # model.fit(X_wide=X_wide, X_tab=X_tab, target=target, n_epochs=10, batch_size=256, val_split=0.2)


        >>> # Ex 3: using train dict and val_split
        >>> # X_train = {'X_wide': X_wide, 'X_tab': X_tab, 'target': y}
        >>> # model.fit(X_train, n_epochs=10, batch_size=256, val_split=0.2)


        >>> # Ex 4: validation using training and validation dicts
        >>> # X_train = {'X_wide': X_wide_tr, 'X_tab': X_tab_tr, 'target': y_tr}
        >>> # X_val = {'X_wide': X_wide_val, 'X_tab': X_tab_val, 'target': y_val}
        >>> # model.fit(X_train=X_train, X_val=X_val n_epochs=10, batch_size=256)

        """

        self.batch_size = batch_size
        train_set, eval_set = self._train_val_split(
            X_wide, X_tab, X_text, X_img, X_train, X_val, val_split, target
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
                        t.set_postfix(loss=train_loss)
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
                                v.set_postfix(loss=val_loss)
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
        self.model.train()

    def predict(
        self,
        X_wide: Optional[np.ndarray] = None,
        X_tab: Optional[np.ndarray] = None,
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
        X_tab: np.ndarray, Optional. Default=None
            Input for the ``deeptabular`` model component.
            See :class:`pytorch_widedeep.preprocessing.TabPreprocessor`
        X_text: np.ndarray, Optional. Default=None
            Input for the ``deeptext`` model component.
            See :class:`pytorch_widedeep.preprocessing.TextPreprocessor`
        X_img : np.ndarray, Optional. Default=None
            Input for the ``deepimage`` model component.
            See :class:`pytorch_widedeep.preprocessing.ImagePreprocessor`
        X_test: Dict[str, np.ndarray], Optional. Default=None
            Testing dataset for the different model components. Keys are
            `'X_wide'`, `'X_tab'`, `'X_text'`, `'X_img'` and `'target'` the values are
            the corresponding matrices.

        """

        preds_l = self._predict(X_wide, X_tab, X_text, X_img, X_test)
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
        X_wide: Optional[np.ndarray] = None,
        X_tab: Optional[np.ndarray] = None,
        X_text: Optional[np.ndarray] = None,
        X_img: Optional[np.ndarray] = None,
        X_test: Optional[Dict[str, np.ndarray]] = None,
    ) -> np.ndarray:
        r"""Returns the predicted probabilities for the test dataset for  binary
        and multiclass methods
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
        format as that of the :obj:`LabelEncoder` Attribute of the class
        :obj:`TabPreprocessor`. See
        :class:`pytorch_widedeep.preprocessing.TabPreprocessor` and
        :class:`pytorch_widedeep.utils.dense_utils.LabelEncder`.

        Parameters
        ----------
        col_name: str,
            Column name of the feature we want to get the embeddings for
        cat_encoding_dict: Dict[str, Dict[str, int]]
            Dictionary containing the categorical encodings, e.g:

        Examples
        --------

        For a series of comprehensive examples please, see the `example
        <https://github.com/jrzaurin/pytorch-widedeep/tree/master/examples>`_.
        folder in the repo

        For completion, here we include a `"fabricated"` example, i.e.
        assuming we have already trained the model, that we have the
        categorical encodings in a dictionary name ``encoding_dict``, and that
        there is a column called `'education'`:

        >>> # model.get_embeddings(col_name='education', cat_encoding_dict=encoding_dict)
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
    ):  # pragma: no cover
        r"""
        Simple wrappup to individually warm up model components
        """
        if self.model.deephead is not None:
            raise ValueError(
                "Currently warming up is only supported without a fully connected 'DeepHead'"
            )
        # This is not the most elegant solution, but is a soluton "in-between"
        # a non elegant one and re-factoring the whole code
        warmer = WarmUp(self.loss_fn, self.metric, self.method, self.verbose)
        warmer.warm_all(self.model.wide, "wide", loader, n_epochs, max_lr)
        warmer.warm_all(self.model.deeptabular, "deeptabular", loader, n_epochs, max_lr)
        if self.model.deeptext:
            if deeptext_gradual:
                warmer.warm_gradual(
                    self.model.deeptext,
                    "deeptext",
                    loader,
                    deeptext_max_lr,
                    deeptext_layers,
                    routine,
                )
            else:
                warmer.warm_all(
                    self.model.deeptext, "deeptext", loader, n_epochs, max_lr
                )
        if self.model.deepimage:
            if deepimage_gradual:
                warmer.warm_gradual(
                    self.model.deepimage,
                    "deepimage",
                    loader,
                    deepimage_max_lr,
                    deepimage_layers,
                    routine,
                )
            else:
                warmer.warm_all(
                    self.model.deepimage, "deepimage", loader, n_epochs, max_lr
                )

    def _lr_scheduler_step(self, step_location: str):  # noqa: C901
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

        if self.metric is not None:
            if self.method == "binary":
                score = self.metric(torch.sigmoid(y_pred), y)
            if self.method == "multiclass":
                score = self.metric(F.softmax(y_pred, dim=1), y)
            return score, avg_loss
        else:
            return None, avg_loss

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
        X_wide: Optional[np.ndarray] = None,
        X_tab: Optional[np.ndarray] = None,
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

    def _get_loss_fn(self, objective, class_weight, custom_loss_function, alpha, gamma):
        if isinstance(class_weight, float):
            class_weight = torch.tensor([1.0 - class_weight, class_weight])
        elif isinstance(class_weight, (tuple, list)):
            class_weight = torch.tensor(class_weight)
        else:
            class_weight = None
        if custom_loss_function is not None:
            return custom_loss_function
        elif self.method != "regression":
            return get_loss_function(objective, weight=class_weight)
        elif "focal_loss" in objective:
            return get_loss_function(objective, alpha=alpha, gamma=gamma)
        else:
            return get_loss_function(objective)

    def _initialize(self, initializers):
        if initializers is not None:
            self.initializer = MultipleInitializer(initializers, verbose=self.verbose)
            self.initializer.apply(self.model)

    def _get_optimizer(self, optimizers):
        if optimizers is not None:
            if isinstance(optimizers, Optimizer):
                optimzer: Union[Optimizer, MultipleOptimizer] = optimizers
            elif isinstance(optimizers, Dict):
                opt_names = list(optimizers.keys())
                mod_names = [
                    n for n, c in self.model.named_children() if n != "loss_fn"
                ]
                for mn in mod_names:
                    assert mn in opt_names, "No optimizer found for {}".format(mn)
                optimzer = MultipleOptimizer(optimizers)
        else:
            optimzer = torch.optim.AdamW(self.model.parameters())  # type: ignore
        return optimzer

    def _get_lr_scheduler(self, lr_schedulers):
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

    def _get_transforms(self, transforms):
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
