import numpy as np
import torch
import torch.nn.functional as F
from tqdm import trange
from torch import nn

from pytorch_widedeep.metrics import Metric, MultipleMetrics
from pytorch_widedeep.wdtypes import *  # noqa: F403

use_cuda = torch.cuda.is_available()


class FineTune:
    r"""
    Fine-tune methods to be applied to the individual model components.

    Note that they can also be used to "fine-tune" those components before
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
        metric: Union[Metric, MultipleMetrics],
        method: str,
        verbose: int,
    ):
        self.loss_fn = loss_fn
        self.metric = metric
        self.method = method
        self.verbose = verbose

    def finetune_all(
        self,
        model: nn.Module,
        model_name: str,
        loader: DataLoader,
        n_epochs: int,
        max_lr: float,
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
        model: `Module``
            ``Module`` object containing one the WideDeep model components (wide,
            deeptabular, deeptext or deepimage)
        model_name: str
            string indicating the model name to access the corresponding parameters.
            One of 'wide', 'deeptabular', 'deeptext' or 'deepimage'
        loader: ``DataLoader``
            Pytorch DataLoader containing the data used to fine-tune
        n_epochs: int
            number of epochs used to fine-tune the model
        max_lr: float
            maximum learning rate value during the triangular cycle.
        """
        if self.verbose:
            print("Training {} for {} epochs".format(model_name, n_epochs))
        model.train()

        optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr / 10.0)  # type: ignore
        step_size_up, step_size_down = self._steps_up_down(len(loader), n_epochs)
        scheduler = torch.optim.lr_scheduler.CyclicLR(
            optimizer,
            base_lr=max_lr / 10.0,
            max_lr=max_lr,
            step_size_up=step_size_up,
            step_size_down=step_size_down,
            cycle_momentum=False,
        )

        self._finetune(
            model, model_name, loader, optimizer, scheduler, n_epochs=n_epochs
        )

    def finetune_gradual(  # noqa: C901
        self,
        model: nn.Module,
        model_name: str,
        loader: DataLoader,
        last_layer_max_lr: float,
        layers: List[nn.Module],
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
        model: ``Module``
           ``Module`` object containing one the WideDeep model components (wide,
           deeptabular, deeptext or deepimage)
        model_name: str
           string indicating the model name to access the corresponding parameters.
           One of 'wide', 'deeptabular', 'deeptext' or 'deepimage'
        loader: ``DataLoader``
           Pytorch DataLoader containing the data to fine-tune with.
        last_layer_max_lr: float
           maximum learning rate value during the triangular cycle for the layer
           closest to the output neuron(s). Deeper layers in 'model' will be trained
           with a gradually descending learning rate. The descending factor is fixed
           and is 2.5
        layers: list
           List of ``Module`` objects containing the layers that will be fine-tuned.
           This must be in *'FINE-TUNE ORDER'*.
        routine: str
           one of 'howard' or 'felbo'
        """
        model.train()

        step_size_up, step_size_down = self._steps_up_down(len(loader))

        original_setup = {}
        for n, p in model.named_parameters():
            original_setup[n] = p.requires_grad
        layers_max_lr = [last_layer_max_lr] + [
            last_layer_max_lr / (2.5 * n) for n in range(1, len(layers))
        ]

        for layer in layers:
            for p in layer.parameters():
                p.requires_grad = False

        if routine == "howard":
            params: List = []
            max_lr: List = []
            base_lr: List = []

        for i, (lr, layer) in enumerate(zip(layers_max_lr, layers)):
            if self.verbose:
                print(
                    "Training {}, layer {} of {}".format(model_name, i + 1, len(layers))
                )
            for p in layer.parameters():
                p.requires_grad = True
            if routine == "felbo":
                params, max_lr, base_lr = layer.parameters(), lr, lr / 10.0  # type: ignore
            elif routine == "howard":
                params += [{"params": layer.parameters(), "lr": lr / 10.0}]
                max_lr += [lr]
                base_lr += [lr / 10.0]
            optimizer = torch.optim.AdamW(params)
            scheduler = torch.optim.lr_scheduler.CyclicLR(
                optimizer,
                base_lr=base_lr,  # type: ignore[arg-type]
                max_lr=max_lr,  # type: ignore
                step_size_up=step_size_up,
                step_size_down=step_size_down,
                cycle_momentum=False,
            )
            self._finetune(model, model_name, loader, optimizer, scheduler)
            if routine == "felbo":
                for p in layer.parameters():
                    p.requires_grad = False

        if routine == "felbo":
            if self.verbose:
                print("Training one last epoch...")
            for layer in layers:
                for p in layer.parameters():
                    p.requires_grad = True
            params, max_lr, base_lr = [], [], []
            for lr, layer in zip(layers_max_lr, layers):
                params += [{"params": layer.parameters(), "lr": lr / 10.0}]
                max_lr += [lr]
                base_lr += [lr / 10.0]
            optimizer = torch.optim.AdamW(params)
            scheduler = torch.optim.lr_scheduler.CyclicLR(
                optimizer,
                base_lr=base_lr,  # type: ignore
                max_lr=max_lr,  # type: ignore
                step_size_up=step_size_up,
                step_size_down=step_size_down,
                cycle_momentum=False,
            )
            self._finetune(model, model_name, loader, optimizer, scheduler)

        for n, p in model.named_parameters():
            p.requires_grad = original_setup[n]

    def _finetune(
        self,
        model: nn.Module,
        model_name: str,
        loader: DataLoader,
        optimizer: Optimizer,
        scheduler: LRScheduler,
        n_epochs: int = 1,
    ):
        r"""
        Standard Pytorch training loop
        """
        steps = len(loader)
        for epoch in range(n_epochs):
            running_loss = 0.0
            with trange(steps, disable=self.verbose != 1) as t:
                for batch_idx, (data, target) in zip(t, loader):
                    t.set_description("epoch %i" % (epoch + 1))
                    X = data[model_name].cuda() if use_cuda else data[model_name]
                    y = (
                        target.view(-1, 1).float()
                        if self.method != "multiclass"
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
        up = round((steps * n_epochs) * 0.1)
        down = (steps * n_epochs) - up
        return up, down
