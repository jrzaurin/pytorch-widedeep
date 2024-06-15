import numpy as np
import torch
import torch.nn.functional as F
from tqdm import trange
from torch import nn

from pytorch_widedeep.metrics import Metric, MultipleMetrics
from pytorch_widedeep.wdtypes import (
    Any,
    List,
    Tuple,
    Union,
    Literal,
    Optional,
    Optimizer,
    DataLoader,
    LRScheduler,
)
from pytorch_widedeep.models._base_wd_model_component import (
    BaseWDModelComponent,
)

use_cuda = torch.cuda.is_available()

WDModel = Union[nn.Module, BaseWDModelComponent]


class FineTune:
    r"""Fine-tune methods to be applied to the individual model components.

    Note that they can also be used to "warm-up" those components before
    the joined training.

    There are 3 fine-tune/warm-up routines available:

    1) Fine-tune all trainable layers at once

    2) Gradual fine-tuning inspired by the work of Felbo et al., 2017

    3) Gradual fine-tuning inspired by the work of Howard & Ruder 2018

    The structure of the code in this class is designed to be instantiated
    within the class WideDeep. This is not ideal, but represents a
    compromise towards implementing a fine-tuning functionality for the
    current overall structure of the package without having to
    re-structure most of the existing code. This will change in future
    releases.

    Parameters
    ----------
    loss_fn: Any
       any function with the same strucure as 'loss_fn' in the class ``Trainer``
    metric: ``Metric`` or ``MultipleMetrics``
       object of class Metric (see Metric in pytorch_widedeep.metrics)
    method: str
       one of 'binary', 'regression' or 'multiclass'
    verbose: Boolean
    """

    def __init__(
        self,
        loss_fn: Any,
        metric: Optional[Union[Metric, MultipleMetrics]],
        method: Literal["binary", "regression", "multiclass", "qregression"],
        verbose: int,
    ):
        self.loss_fn = loss_fn
        self.metric = metric
        self.method = method
        self.verbose = verbose

    def finetune_all(
        self,
        model: Union[WDModel, nn.ModuleList],
        model_name: str,
        loader: DataLoader,
        n_epochs: int,
        max_lr: Union[float, List[float]],
    ):
        r"""Fine-tune/warm-up all trainable layers in a model using a one cyclic
        learning rate with a triangular pattern. This is refereed as Slanted
        Triangular learing rate in Jeremy Howard & Sebastian Ruder 2018
        (https://arxiv.org/abs/1801.06146). The cycle is described as follows:

        1) The learning rate will gradually increase for 10% of the training steps
            from max_lr/10 to max_lr.

        2) It will then gradually decrease to max_lr/10 for the remaining 90% of the
            steps.

        The optimizer used in the process is AdamW

        Parameters:
        ----------
        model: WDModel or nn.ModuleList
            ``Module`` object containing one the WideDeep model components (wide,
            deeptabular, deeptext or deepimage)
        model_name: str
            string indicating the model name to access the corresponding parameters.
            One of 'wide', 'deeptabular', 'deeptext' or 'deepimage'
        loader: ``DataLoader``
            Pytorch DataLoader containing the data used to fine-tune
        n_epochs: int
            number of epochs used to fine-tune the model
        max_lr: float or List[float]
            maximum learning rate value during the triangular cycle.
        """
        if self.verbose:
            print("Training {} for {} epochs".format(model_name, n_epochs))

        if isinstance(model, nn.ModuleList):

            for i, _model in enumerate(model):

                if isinstance(max_lr, list):
                    _max_lr = max_lr[i]
                else:
                    _max_lr = max_lr

                _model.train()

                self.finetune_one(_model, model_name, loader, n_epochs, _max_lr, idx=i)

        else:
            assert isinstance(max_lr, float)

            self.finetune_one(model, model_name, loader, n_epochs, max_lr)

    def finetune_gradual(  # noqa: C901
        self,
        model: Union[WDModel, nn.ModuleList],
        model_name: str,
        loader: DataLoader,
        last_layer_max_lr: Union[float, List[float]],
        layers: Union[List[nn.Module], List[List[nn.Module]]],
        routine: str,
    ):
        r"""Fine-tune/warm-up certain layers within the model following a
        gradual fine-tune routine. The approaches implemented in this method are
        based on fine-tuning routines described in the the work of Felbo et
        al., 2017 in their DeepEmoji paper (https://arxiv.org/abs/1708.00524)
        and Howard & Sebastian Ruder 2018 ULMFit paper
        (https://arxiv.org/abs/1801.06146).

        A one cycle triangular learning rate is used. In both Felbo's and
        Howard's routines a gradually decreasing learning rate is used as we
        go deeper into the network. The 'closest' layer to the output
        neuron(s) will use a maximum learning rate of 'last_layer_max_lr'. The
        learning rate will then decrease by a factor of 2.5 per layer

        1) The 'Felbo' routine: train the first layer in 'layers' for one
           epoch. Then train the next layer in 'layers' for one epoch freezing
           the already trained up layer(s). Repeat untill all individual layers
           are trained. Then, train one last epoch with all trained/fine-tuned
           layers trainable

        2) The 'Howard' routine: fine-tune the first layer in 'layers' for one
           epoch. Then traine the next layer in the model for one epoch while
           keeping the already trained up layer(s) trainable. Repeat.

        Parameters:
        ----------
        model: WDModel or nn.ModuleList
           ``Module`` object containing one the WideDeep model components (wide,
           deeptabular, deeptext or deepimage)
        model_name: str
           string indicating the model name to access the corresponding parameters.
           One of 'wide', 'deeptabular', 'deeptext' or 'deepimage'
        loader: ``DataLoader``
           Pytorch DataLoader containing the data to fine-tune with.
        last_layer_max_lr: float or List[float]
           maximum learning rate value during the triangular cycle for the layer
           closest to the output neuron(s). Deeper layers in 'model' will be trained
           with a gradually descending learning rate. The descending factor is fixed
           and is 2.5
        layers: List[nn.Module] or List[List[nn.Module]]
           List of ``Module`` objects containing the layers that will be fine-tuned.
           This must be in *'FINE-TUNE ORDER'*.
        routine: str
           one of 'howard' or 'felbo'
        """

        if isinstance(model, nn.ModuleList):

            for i, _model in enumerate(model):

                assert isinstance(layers[i], list)

                self._finetune_gradual_one(
                    _model,
                    model_name,
                    loader,
                    (
                        last_layer_max_lr[i]
                        if isinstance(last_layer_max_lr, list)
                        else last_layer_max_lr
                    ),
                    layers[i],  # type: ignore[arg-type]
                    routine,
                    idx=i,
                )
        else:

            assert isinstance(layers, list)
            assert isinstance(last_layer_max_lr, float)

            self._finetune_gradual_one(
                model,
                model_name,
                loader,
                last_layer_max_lr,
                layers,  # type: ignore[arg-type]
                routine,
            )

    def _finetune_howard(
        self,
        layers: List[nn.Module],
        layers_max_lr: List[float],
        step_size_up: int,
        step_size_down: int,
        model: WDModel,
        model_name: str,
        loader: DataLoader,
        idx: Optional[int] = None,
    ):

        for layer in layers:
            for p in layer.parameters():
                p.requires_grad = False

        params, max_lr, base_lr = [], [], []
        for i, (lr, layer) in enumerate(zip(layers_max_lr, layers)):
            if self.verbose:
                print(
                    "Training {}, layer {} of {}".format(model_name, i + 1, len(layers))
                )

            for p in layer.parameters():
                p.requires_grad = True

            params += [{"params": layer.parameters(), "lr": lr / 10.0}]
            max_lr += [lr]
            base_lr += [lr / 10.0]

            optimizer = torch.optim.AdamW(params)

            scheduler = torch.optim.lr_scheduler.CyclicLR(
                optimizer,
                base_lr=base_lr,
                max_lr=max_lr,
                step_size_up=step_size_up,
                step_size_down=step_size_down,
                cycle_momentum=False,
            )

            self._train(model, model_name, loader, optimizer, scheduler, idx=idx)

    def finetune_felbo(  # noqa: C901
        self,
        layers: List[nn.Module],
        layers_max_lr: List[float],
        step_size_up: int,
        step_size_down: int,
        model: WDModel,
        model_name: str,
        loader: DataLoader,
        idx: Optional[int] = None,
    ):

        for layer in layers:
            for p in layer.parameters():
                p.requires_grad = False

        for i, (lr, layer) in enumerate(zip(layers_max_lr, layers)):
            if self.verbose:
                print(
                    "Training {}, layer {} of {}".format(model_name, i + 1, len(layers))
                )

            for p in layer.parameters():
                p.requires_grad = True

            params, max_lr, base_lr = layer.parameters(), lr, lr / 10.0

            optimizer = torch.optim.AdamW(params)

            scheduler = torch.optim.lr_scheduler.CyclicLR(
                optimizer,
                base_lr=base_lr,
                max_lr=max_lr,
                step_size_up=step_size_up,
                step_size_down=step_size_down,
                cycle_momentum=False,
            )

            self._train(model, model_name, loader, optimizer, scheduler, idx=idx)

            for p in layer.parameters():
                p.requires_grad = False

            if self.verbose:
                print("Training one last epoch...")

            for layer in layers:
                for p in layer.parameters():
                    p.requires_grad = True

            params_, max_lr_, base_lr_ = [], [], []
            for lr, layer in zip(layers_max_lr, layers):
                params_ += [{"params": layer.parameters(), "lr": lr / 10.0}]
                max_lr_ += [lr]
                base_lr_ += [lr / 10.0]

            optimizer = torch.optim.AdamW(params_)

            scheduler = torch.optim.lr_scheduler.CyclicLR(
                optimizer,
                base_lr=base_lr_,
                max_lr=max_lr_,
                step_size_up=step_size_up,
                step_size_down=step_size_down,
                cycle_momentum=False,
            )

            self._train(model, model_name, loader, optimizer, scheduler, idx=idx)

    def finetune_one(
        self,
        model: WDModel,
        model_name: str,
        loader: DataLoader,
        n_epochs: int,
        max_lr: float,
        idx: Optional[int] = None,
    ):

        model.train()

        optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr / 10.0)
        step_size_up, step_size_down = self._steps_up_down(len(loader), n_epochs)
        scheduler = torch.optim.lr_scheduler.CyclicLR(
            optimizer,
            base_lr=max_lr / 10.0,
            max_lr=max_lr,
            step_size_up=step_size_up,
            step_size_down=step_size_down,
            cycle_momentum=False,
        )

        self._train(
            model, model_name, loader, optimizer, scheduler, n_epochs=n_epochs, idx=idx
        )

    def _finetune_gradual_one(
        self,
        model: WDModel,
        model_name: str,
        loader: DataLoader,
        last_layer_max_lr: float,
        layers: List[nn.Module],
        routine: str,
        idx: Optional[int] = None,
    ):

        original_setup = {}
        for n, p in model.named_parameters():
            original_setup[n] = p.requires_grad

        model.train()

        layers_max_lr = [last_layer_max_lr] + [
            last_layer_max_lr / (2.5 * n) for n in range(1, len(layers))
        ]

        step_size_up, step_size_down = self._steps_up_down(len(loader))

        if routine == "howard":
            self._finetune_howard(
                layers,
                layers_max_lr,
                step_size_up,
                step_size_down,
                model,
                model_name,
                loader,
                idx=idx,
            )
        elif routine == "felbo":
            self.finetune_felbo(
                layers,
                layers_max_lr,
                step_size_up,
                step_size_down,
                model,
                model_name,
                loader,
                idx=idx,
            )
        else:
            raise ValueError(
                "routine must be one of 'howard' or 'felbo'. Got {}".format(routine)
            )

        for n, p in model.named_parameters():
            p.requires_grad = original_setup[n]

    def _train(  # noqa: C901
        self,
        model: WDModel,
        model_name: str,
        loader: DataLoader,
        optimizer: Optimizer,
        scheduler: LRScheduler,
        n_epochs: int = 1,
        idx: Optional[int] = None,
    ):
        r"""
        Standard Pytorch training loop
        """
        steps = len(loader)
        for epoch in range(n_epochs):
            running_loss = 0.0
            with trange(steps, disable=self.verbose != 1) as t:
                for batch_idx, packed_data in zip(t, loader):
                    if idx is not None:
                        t.set_description(f"epoch {epoch} for {model_name} {idx}")
                    else:
                        t.set_description("epoch %i" % (epoch + 1))

                    try:
                        data, target, _ = packed_data
                    except ValueError:
                        data, target = packed_data

                    if idx is not None:
                        X = (
                            data[model_name][idx].cuda()
                            if use_cuda
                            else data[model_name][idx]
                        )
                    else:
                        X = data[model_name].cuda() if use_cuda else data[model_name]

                    y = (
                        target.view(-1, 1).float()
                        if self.method not in ["multiclass", "qregression"]
                        else target
                    )
                    y = y.cuda() if use_cuda else y

                    optimizer.zero_grad()
                    y_pred = model(X)
                    loss = self.loss_fn(y_pred, y)
                    loss.backward()
                    optimizer.step()
                    scheduler.step()

                    running_loss += loss.item()
                    avg_loss = running_loss / (batch_idx + 1)

                    if self.metric is not None:
                        if self.method == "regression":
                            score = self.metric(y_pred, y)
                        if self.method == "binary":
                            score = self.metric(torch.sigmoid(y_pred), y)
                        if self.method == "qregression":
                            score = self.metric(y_pred, y)
                        if self.method == "multiclass":
                            score = self.metric(F.softmax(y_pred, dim=1), y)
                        t.set_postfix(
                            metrics={k: np.round(v, 4) for k, v in score.items()},
                            loss=avg_loss,
                        )
                    else:
                        t.set_postfix(loss=avg_loss)

    def _steps_up_down(self, steps: int, n_epochs: int = 1) -> Tuple[int, int]:
        r"""
        Calculate the number of steps up and down during the one cycle fine-tune for a
        given number of epochs

        Parameters:
        ----------
        steps: int
            steps per epoch
        n_epochs: int, default=1
            number of fine-tune epochs

        Returns:
        -------
        up, down: Tuple, int
            number of steps increasing/decreasing the learning rate during the cycle
        """
        # up = round((steps * n_epochs) * 0.1)
        up = max([round((steps * n_epochs) * 0.1), 1])
        down = (steps * n_epochs) - up
        return up, down
