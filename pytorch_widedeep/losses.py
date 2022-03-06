import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_widedeep.wdtypes import *  # noqa: F403

use_cuda = torch.cuda.is_available()


class MSELoss(nn.Module):
    r"""Mean square error loss adjusted for the possibility of using Label Smooth
    Distribution (LDS)

    LDS is based on `Delving into Deep Imbalanced Regression
    <https://arxiv.org/abs/2102.09554>`_. and their `implementation
    <https://github.com/YyzHarry/imbalanced-regression>`_
    """

    def __init__(self):
        super().__init__()

    def forward(
        self, input: Tensor, target: Tensor, lds_weight: Optional[Tensor] = None
    ) -> Tensor:
        r"""
        Parameters
        ----------
        input: Tensor
            Input tensor with predictions
        target: Tensor
            Target tensor with the actual values
        lds_weight: Tensor, Optional
            If we choose to use LDS this is the tensor of weights that will
            multiply the loss value.

        Examples
        --------
        >>> import torch
        >>> from pytorch_widedeep.losses import MSELoss
        >>>
        >>> target = torch.tensor([1, 1.2, 0, 2]).view(-1, 1)
        >>> input = torch.tensor([0.6, 0.7, 0.3, 0.8]).view(-1, 1)
        >>> lds_weight = torch.tensor([0.1, 0.2, 0.3, 0.4]).view(-1, 1)
        >>> MSELoss()(input, target, lds_weight)
        tensor(0.1673)
        """
        loss = (input - target) ** 2
        if lds_weight is not None:
            loss *= lds_weight
        return torch.mean(loss)


class MSLELoss(nn.Module):
    r"""Mean square log error loss adjusted for the possibility of using Label
    Smooth Distribution (LDS)

    LDS is based on `Delving into Deep Imbalanced Regression
    <https://arxiv.org/abs/2102.09554>`_. and their `implementation
    <https://github.com/YyzHarry/imbalanced-regression>`_
    """

    def __init__(self):
        super().__init__()

    def forward(
        self, input: Tensor, target: Tensor, lds_weight: Optional[Tensor] = None
    ) -> Tensor:
        r"""
        Parameters
        ----------
        input: Tensor
            Input tensor with predictions (not probabilities)
        target: Tensor
            Target tensor with the actual classes
        lds_weight: Tensor, Optional
            If we choose to use LDS this is the tensor of weights that will
            multiply the loss value.

        Examples
        --------
        >>> import torch
        >>> from pytorch_widedeep.losses import MSLELoss
        >>>
        >>> target = torch.tensor([1, 1.2, 0, 2]).view(-1, 1)
        >>> input = torch.tensor([0.6, 0.7, 0.3, 0.8]).view(-1, 1)
        >>> lds_weight = torch.tensor([0.1, 0.2, 0.3, 0.4]).view(-1, 1)
        >>> MSLELoss()(input, target, lds_weight)
        tensor(0.0358)
        """
        assert (
            input.min() >= 0
        ), """All input values must be >=0, if your model is predicting
            values <0 try to enforce positive values by activation function
            on last layer with `trainer.enforce_positive_output=True`"""
        assert target.min() >= 0, "All target values must be >=0"

        loss = (torch.log(input + 1) - torch.log(target + 1)) ** 2
        if lds_weight is not None:
            loss *= lds_weight
        return torch.mean(loss)


class RMSELoss(nn.Module):
    r"""Root mean square error loss adjusted for the possibility of using Label
    Smooth Distribution (LDS)

    LDS is based on `Delving into Deep Imbalanced Regression
    <https://arxiv.org/abs/2102.09554>`_. and their `implementation
    <https://github.com/YyzHarry/imbalanced-regression>`_
    """

    def __init__(self):
        super().__init__()

    def forward(
        self, input: Tensor, target: Tensor, lds_weight: Optional[Tensor] = None
    ) -> Tensor:
        r"""
        Parameters
        ----------
        input: Tensor
            Input tensor with predictions (not probabilities)
        target: Tensor
            Target tensor with the actual classes
        lds_weight: Tensor, Optional
            If we choose to use LDS this is the tensor of weights that will
            multiply the loss value.

        Examples
        --------
        >>> import torch
        >>> from pytorch_widedeep.losses import RMSELoss
        >>>
        >>> target = torch.tensor([1, 1.2, 0, 2]).view(-1, 1)
        >>> input = torch.tensor([0.6, 0.7, 0.3, 0.8]).view(-1, 1)
        >>> lds_weight = torch.tensor([0.1, 0.2, 0.3, 0.4]).view(-1, 1)
        >>> RMSELoss()(input, target, lds_weight)
        tensor(0.4090)
        """
        loss = (input - target) ** 2
        if lds_weight is not None:
            loss *= lds_weight
        return torch.sqrt(torch.mean(loss))


class RMSLELoss(nn.Module):
    r"""Root mean square log error loss adjusted for the possibility of using Label
    Smooth Distribution (LDS)

    LDS is based on `Delving into Deep Imbalanced Regression
    <https://arxiv.org/abs/2102.09554>`_. and their `implementation
    <https://github.com/YyzHarry/imbalanced-regression>`_
    """

    def __init__(self):
        super().__init__()

    def forward(
        self, input: Tensor, target: Tensor, lds_weight: Optional[Tensor] = None
    ) -> Tensor:
        r"""
        Parameters
        ----------
        input: Tensor
            Input tensor with predictions (not probabilities)
        target: Tensor
            Target tensor with the actual classes
        lds_weight: Tensor, Optional
            If we choose to use LDS this is the tensor of weights that will
            multiply the loss value.

        Examples
        --------
        >>> import torch
        >>> from pytorch_widedeep.losses import RMSLELoss
        >>>
        >>> target = torch.tensor([1, 1.2, 0, 2]).view(-1, 1)
        >>> input = torch.tensor([0.6, 0.7, 0.3, 0.8]).view(-1, 1)
        >>> lds_weight = torch.tensor([0.1, 0.2, 0.3, 0.4]).view(-1, 1)
        >>> RMSELoss()(input, target, lds_weight)
        tensor(0.4090)
        """
        assert (
            input.min() >= 0
        ), """All input values must be >=0, if your model is predicting
            values <0 try to enforce positive values by activation function
            on last layer with `trainer.enforce_positive_output=True`"""
        assert target.min() >= 0, "All target values must be >=0"

        loss = (torch.log(input + 1) - torch.log(target + 1)) ** 2
        if lds_weight is not None:
            loss *= lds_weight
        return torch.sqrt(torch.mean(loss))


class QuantileLoss(nn.Module):
    r"""Quantile loss defined as:

        :math:`Loss = max(q \times (y-y_{pred}), (1-q) \times (y_{pred}-y))`

    All credits go to the implementation at `pytorch-forecasting
    <https://pytorch-forecasting.readthedocs.io/en/latest/_modules/pytorch_forecasting/metrics.html#QuantileLoss>`_ .

    Parameters
    ----------
    quantiles: List, default = [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]
        List of quantiles
    """

    def __init__(
        self,
        quantiles: List[float] = [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98],
    ):
        super().__init__()
        self.quantiles = quantiles

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        r"""
        Parameters
        ----------
        input: Tensor
            Input tensor with predictions
        target: Tensor
            Target tensor with the actual values

        Examples
        --------
        >>> import torch
        >>>
        >>> from pytorch_widedeep.losses import QuantileLoss
        >>>
        >>> # REGRESSION
        >>> target = torch.tensor([[0.6, 1.5]]).view(-1, 1)
        >>> input = torch.tensor([[.1, .2,], [.4, .5]])
        >>> qloss = QuantileLoss([0.25, 0.75])
        >>> qloss(input, target)
        tensor(0.3625)
        """

        assert input.shape == torch.Size([target.shape[0], len(self.quantiles)]), (
            "The input and target have inconsistent shape. The dimension of the prediction "
            "of the model that is using QuantileLoss must be equal to number of quantiles, "
            f"i.e. {len(self.quantiles)}."
        )
        target = target.view(-1, 1).float()
        losses = []
        for i, q in enumerate(self.quantiles):
            errors = target - input[..., i]
            losses.append(torch.max((q - 1) * errors, q * errors).unsqueeze(-1))

        loss = torch.cat(losses, dim=2)

        return torch.mean(loss)


class FocalLoss(nn.Module):
    r"""Implementation of the `focal loss
    <https://arxiv.org/pdf/1708.02002.pdf>`_ for both binary and
    multiclass classification

    :math:`FL(p_t) = \alpha (1 - p_t)^{\gamma} log(p_t)`

    where, for a case of a binary classification problem

    :math:`\begin{equation} p_t= \begin{cases}p, & \text{if $y=1$}.\\1-p, & \text{otherwise}. \end{cases} \end{equation}`

    Parameters
    ----------
    alpha: float
        Focal Loss ``alpha`` parameter
    gamma: float
        Focal Loss ``gamma`` parameter
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def _get_weight(self, p: Tensor, t: Tensor) -> Tensor:
        pt = p * t + (1 - p) * (1 - t)  # type: ignore
        w = self.alpha * t + (1 - self.alpha) * (1 - t)  # type: ignore
        return (w * (1 - pt).pow(self.gamma)).detach()  # type: ignore

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        r"""
        Parameters
        ----------
        input: Tensor
            Input tensor with predictions (not probabilities)
        target: Tensor
            Target tensor with the actual classes

        Examples
        --------
        >>> import torch
        >>>
        >>> from pytorch_widedeep.losses import FocalLoss
        >>>
        >>> # BINARY
        >>> target = torch.tensor([0, 1, 0, 1]).view(-1, 1)
        >>> input = torch.tensor([[0.6, 0.7, 0.3, 0.8]]).t()
        >>> FocalLoss()(input, target)
        tensor(0.1762)
        >>>
        >>> # MULTICLASS
        >>> target = torch.tensor([1, 0, 2]).view(-1, 1)
        >>> input = torch.tensor([[0.2, 0.5, 0.3], [0.8, 0.1, 0.1], [0.7, 0.2, 0.1]])
        >>> FocalLoss()(input, target)
        tensor(0.2573)
        """
        input_prob = torch.sigmoid(input)
        if input.size(1) == 1:
            input_prob = torch.cat([1 - input_prob, input_prob], axis=1)  # type: ignore
            num_class = 2
        else:
            num_class = input_prob.size(1)
        binary_target = torch.eye(num_class)[target.squeeze().long()]
        if use_cuda:
            binary_target = binary_target.cuda()
        binary_target = binary_target.contiguous()
        weight = self._get_weight(input_prob, binary_target)

        return F.binary_cross_entropy(
            input_prob, binary_target, weight, reduction="mean"
        )


class BayesianRegressionLoss(nn.Module):
    r"""log Gaussian loss as specified in the original publication 'Weight
    Uncertainty in Neural Networks'
    Currently we do not use this loss as is proportional to the
    ``BayesianSELoss`` and the latter does not need a scale/noise_tolerance
    param
    """

    def __init__(self, noise_tolerance: float):
        super().__init__()
        self.noise_tolerance = noise_tolerance

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return (
            -torch.distributions.Normal(input, self.noise_tolerance)
            .log_prob(target)
            .sum()
        )


class BayesianSELoss(nn.Module):
    r"""Squared Loss (log Gaussian) for the case of a regression as specified in
    the original publication `Weight Uncertainty in Neural Networks
    <https://arxiv.org/abs/1505.05424>`_
    """

    def __init__(self):
        super().__init__()

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        r"""
        Parameters
        ----------
        input: Tensor
            Input tensor with predictions (not probabilities)
        target: Tensor
            Target tensor with the actual classes
        Examples
        --------
        >>> import torch
        >>> from pytorch_widedeep.losses import BayesianSELoss
        >>> target = torch.tensor([1, 1.2, 0, 2]).view(-1, 1)
        >>> input = torch.tensor([0.6, 0.7, 0.3, 0.8]).view(-1, 1)
        >>> BayesianSELoss()(input, target)
        tensor(0.9700)
        """
        return (0.5 * (input - target) ** 2).sum()


class TweedieLoss(nn.Module):
    """
    Tweedie loss for extremely unbalanced zero-inflated data

    All credits go to Wenbo Shi.
    See `this post <https://towardsdatascience.com/tweedie-loss-function-for-right-skewed-data-2c5ca470678f>`_
    and the `original publication <https://arxiv.org/abs/1811.10192>`_ for details
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        input: Tensor,
        target: Tensor,
        lds_weight: Optional[Tensor] = None,
        p: float = 1.5,
    ) -> Tensor:
        r"""
        Parameters
        ----------
        input: Tensor
            Input tensor with predictions
        target: Tensor
            Target tensor with the actual values
        lds_weight: Tensor, Optional
            If we choose to use LDS this is the tensor of weights that will
            multiply the loss value.
        p: float, default = 1.5
            the power to be used to compute the loss. See the original
            publication for details

        Examples
        --------
        >>> import torch
        >>> from pytorch_widedeep.losses import TweedieLoss
        >>>
        >>> target = torch.tensor([1, 1.2, 0, 2]).view(-1, 1)
        >>> input = torch.tensor([0.6, 0.7, 0.3, 0.8]).view(-1, 1)
        >>> lds_weight = torch.tensor([0.1, 0.2, 0.3, 0.4]).view(-1, 1)
        >>> TweedieLoss()(input, target, lds_weight)
        tensor(1.0386)
        """

        assert (
            input.min() > 0
        ), """All input values must be >=0, if your model is predicting
            values <0 try to enforce positive values by activation function
            on last layer with `trainer.enforce_positive_output=True`"""
        assert target.min() >= 0, "All target values must be >=0"
        loss = -target * torch.pow(input, 1 - p) / (1 - p) + torch.pow(input, 2 - p) / (
            2 - p
        )
        if lds_weight is not None:
            loss *= lds_weight

        return torch.mean(loss)


class ZILNLoss(nn.Module):
    r"""Adjusted implementation of the Zero Inflated LogNormal Loss

    See the `paper <https://arxiv.org/pdf/1912.07753.pdf>`_ and
    the corresponding `code
    <https://github.com/google/lifetime_value/blob/master/lifetime_value/zero_inflated_lognormal.py>`_
    """

    def __init__(self):
        super().__init__()

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        r"""
        Parameters
        ----------
        input: Tensor
            Input tensor with predictions with spape (N,3), where N is the batch size
        target: Tensor
            Target tensor with the actual target values

        Examples
        --------
        >>> import torch
        >>> from pytorch_widedeep.losses import ZILNLoss
        >>>
        >>> target = torch.tensor([[0., 1.5]]).view(-1, 1)
        >>> input = torch.tensor([[.1, .2, .3], [.4, .5, .6]])
        >>> ZILNLoss()(input, target)
        tensor(1.3114)
        """
        positive = target > 0
        positive = positive.float()

        assert input.shape == torch.Size([target.shape[0], 3]), (
            "Wrong shape of the 'input' tensor. The pred_dim of the "
            "model that is using ZILNLoss must be equal to 3."
        )

        positive_input = input[..., :1]

        classification_loss = F.binary_cross_entropy_with_logits(
            positive_input, positive, reduction="none"
        ).flatten()

        loc = input[..., 1:2]

        # when using max the two input tensors (input and other) have to be of
        # the same type
        max_input = F.softplus(input[..., 2:])
        max_other = torch.sqrt(torch.Tensor([torch.finfo(torch.double).eps])).type(
            max_input.type()
        )
        scale = torch.max(max_input, max_other)
        safe_labels = positive * target + (1 - positive) * torch.ones_like(target)

        regression_loss = -torch.mean(
            positive
            * torch.distributions.log_normal.LogNormal(loc=loc, scale=scale).log_prob(
                safe_labels
            ),
            dim=-1,
        )

        return torch.mean(classification_loss + regression_loss)


class L1Loss(nn.Module):
    r"""L1 loss adjusted for the possibility of using Label Smooth
    Distribution (LDS)

    Based on `Delving into Deep Imbalanced Regression
    <https://arxiv.org/abs/2102.09554>`_. and their `implementation
    <https://github.com/YyzHarry/imbalanced-regression>`_
    """

    def __init__(self):
        super().__init__()

    def forward(
        self, input: Tensor, target: Tensor, lds_weight: Optional[Tensor] = None
    ) -> Tensor:
        r"""
        Parameters
        ----------
        input: Tensor
            Input tensor with predictions
        target: Tensor
            Target tensor with the actual values
        lds_weight: Tensor, Optional
            If we choose to use LDS this is the tensor of weights that will
            multiply the loss value.

        Examples
        --------
        >>> import torch
        >>>
        >>> from pytorch_widedeep.losses import L1Loss
        >>>
        >>> target = torch.tensor([1, 1.2, 0, 2]).view(-1, 1)
        >>> input = torch.tensor([0.6, 0.7, 0.3, 0.8]).view(-1, 1)
        >>> L1Loss()(input, target)
        tensor(0.6000)
        """
        loss = F.l1_loss(input, target, reduction="none")
        if lds_weight is not None:
            loss *= lds_weight
        return torch.mean(loss)


class FocalR_L1Loss(nn.Module):
    r"""Focal-R L1 loss

    Based on `Delving into Deep Imbalanced Regression
    <https://arxiv.org/abs/2102.09554>`_ and their `implementation
    <https://github.com/YyzHarry/imbalanced-regression>`_

    Parameters
    ----------
    beta: float
        Focal Loss ``beta`` parameter in their implementation
    gamma: float
        Focal Loss ``gamma`` parameter
    activation_fn: str, default = "sigmoid"
        Activation function to be used during the computation of the loss.
        Possible values are `'sigmoid'` and `'tanh'`. See the original
        publication for details.
    """

    def __init__(
        self,
        beta: float = 0.2,
        gamma: float = 1.0,
        activation_fn: Literal["sigmoid", "tanh"] = "sigmoid",
    ):
        super().__init__()
        self.beta = beta
        self.gamma = gamma
        self.activation_fn = activation_fn

    def forward(
        self,
        input: Tensor,
        target: Tensor,
        lds_weight: Optional[Tensor] = None,
    ) -> Tensor:
        r"""
        Parameters
        ----------
        input: Tensor
            Input tensor with predictions (not probabilities)
        target: Tensor
            Target tensor with the actual classes
        lds_weight: Tensor, Optional
            If we choose to use LDS this is the tensor of weights that will
            multiply the loss value.

        Examples
        --------
        >>> import torch
        >>>
        >>> from pytorch_widedeep.losses import FocalR_L1Loss
        >>>
        >>> target = torch.tensor([1, 1.2, 0, 2]).view(-1, 1)
        >>> input = torch.tensor([0.6, 0.7, 0.3, 0.8]).view(-1, 1)
        >>> FocalR_L1Loss()(input, target)
        tensor(0.0483)
        """
        loss = F.l1_loss(input, target, reduction="none")
        if self.activation_fn == "tanh":
            loss *= (torch.tanh(self.beta * torch.abs(input - target))) ** self.gamma
        elif self.activation_fn == "sigmoid":
            loss *= (
                2 * torch.sigmoid(self.beta * torch.abs(input - target)) - 1
            ) ** self.gamma
        else:
            ValueError(
                "Incorrect activation function value - must be in ['sigmoid', 'tanh']"
            )
        if lds_weight is not None:
            loss *= lds_weight
        return torch.mean(loss)


class FocalR_MSELoss(nn.Module):
    r"""Focal-R MSE loss

    Based on `Delving into Deep Imbalanced Regression
    <https://arxiv.org/abs/2102.09554>`_ and their `implementation
    <https://github.com/YyzHarry/imbalanced-regression>`_

    Parameters
    ----------
    beta: float
        Focal Loss ``beta`` parameter in their implementation
    gamma: float
        Focal Loss ``gamma`` parameter
    activation_fn: str, default = "sigmoid"
        Activation function to be used during the computation of the loss.
        Possible values are `'sigmoid'` and `'tanh'`. See the original
        publication for details.
    """

    def __init__(
        self,
        beta: float = 0.2,
        gamma: float = 1.0,
        activation_fn: Literal["sigmoid", "tanh"] = "sigmoid",
    ):
        super().__init__()
        self.beta = beta
        self.gamma = gamma
        self.activation_fn = activation_fn

    def forward(
        self,
        input: Tensor,
        target: Tensor,
        lds_weight: Optional[Tensor] = None,
    ) -> Tensor:
        r"""
        Parameters
        ----------
        input: Tensor
            Input tensor with predictions (not probabilities)
        target: Tensor
            Target tensor with the actual classes
        lds_weight: Tensor, Optional
            If we choose to use LDS this is the tensor of weights that will
            multiply the loss value.

        Examples
        --------
        >>> import torch
        >>>
        >>> from pytorch_widedeep.losses import FocalR_MSELoss
        >>>
        >>> target = torch.tensor([1, 1.2, 0, 2]).view(-1, 1)
        >>> input = torch.tensor([0.6, 0.7, 0.3, 0.8]).view(-1, 1)
        >>> FocalR_MSELoss()(input, target)
        tensor(0.0539)
        """
        loss = (input - target) ** 2
        if self.activation_fn == "tanh":
            loss *= (torch.tanh(self.beta * torch.abs(input - target))) ** self.gamma
        elif self.activation_fn == "sigmoid":
            loss *= (
                2 * torch.sigmoid(self.beta * torch.abs((input - target) ** 2)) - 1
            ) ** self.gamma
        else:
            ValueError(
                "Incorrect activation function value - must be in ['sigmoid', 'tanh']"
            )
        if lds_weight is not None:
            loss *= lds_weight
        return torch.mean(loss)


class FocalR_RMSELoss(nn.Module):
    r"""Focal-R RMSE loss

    Based on `Delving into Deep Imbalanced Regression
    <https://arxiv.org/abs/2102.09554>`_ and their `implementation
    <https://github.com/YyzHarry/imbalanced-regression>`_

    Parameters
    ----------
    beta: float
        Focal Loss ``beta`` parameter in their implementation
    gamma: float
        Focal Loss ``gamma`` parameter
    activation_fn: str, default = "sigmoid"
        Activation function to be used during the computation of the loss.
        Possible values are `'sigmoid'` and `'tanh'`. See the original
        publication for details.
    """

    def __init__(
        self,
        beta: float = 0.2,
        gamma: float = 1.0,
        activation_fn: Literal["sigmoid", "tanh"] = "sigmoid",
    ):
        super().__init__()
        self.beta = beta
        self.gamma = gamma
        self.activation_fn = activation_fn

    def forward(
        self,
        input: Tensor,
        target: Tensor,
        lds_weight: Optional[Tensor] = None,
    ) -> Tensor:
        r"""
        Parameters
        ----------
        input: Tensor
            Input tensor with predictions (not probabilities)
        target: Tensor
            Target tensor with the actual classes
        lds_weight: Tensor, Optional
            If we choose to use LDS this is the tensor of weights that will
            multiply the loss value.

        Examples
        --------
        >>> import torch
        >>>
        >>> from pytorch_widedeep.losses import FocalR_RMSELoss
        >>>
        >>> target = torch.tensor([1, 1.2, 0, 2]).view(-1, 1)
        >>> input = torch.tensor([0.6, 0.7, 0.3, 0.8]).view(-1, 1)
        >>> FocalR_RMSELoss()(input, target)
        tensor(0.2321)
        """
        loss = (input - target) ** 2
        if self.activation_fn == "tanh":
            loss *= (torch.tanh(self.beta * torch.abs(input - target))) ** self.gamma
        elif self.activation_fn == "sigmoid":
            loss *= (
                2 * torch.sigmoid(self.beta * torch.abs((input - target) ** 2)) - 1
            ) ** self.gamma
        else:
            ValueError(
                "Incorrect activation function value - must be in ['sigmoid', 'tanh']"
            )
        if lds_weight is not None:
            loss *= lds_weight
        return torch.sqrt(torch.mean(loss))


class HuberLoss(nn.Module):
    r"""Hubbler Loss

    Based on `Delving into Deep Imbalanced Regression
    <https://arxiv.org/abs/2102.09554>`_ and their `implementation
    <https://github.com/YyzHarry/imbalanced-regression>`_
    """

    def __init__(self, beta: float = 0.2):
        super().__init__()
        self.beta = beta

    def forward(
        self,
        input: Tensor,
        target: Tensor,
        lds_weight: Optional[Tensor] = None,
    ) -> Tensor:
        r"""
        Parameters
        ----------
        input: Tensor
            Input tensor with predictions (not probabilities)
        target: Tensor
            Target tensor with the actual classes
        lds_weight: Tensor, Optional
            If we choose to use LDS this is the tensor of weights that will
            multiply the loss value.
        activation_fn: str, default = "sigmoid"
            Activation function to be used during the computation of the loss.
            Possible values are `'sigmoid'` and `'tanh'`. See the original
            publication for details.

        Examples
        --------
        >>> import torch
        >>>
        >>> from pytorch_widedeep.losses import HuberLoss
        >>>
        >>> target = torch.tensor([1, 1.2, 0, 2]).view(-1, 1)
        >>> input = torch.tensor([0.6, 0.7, 0.3, 0.8]).view(-1, 1)
        >>> HuberLoss()(input, target)
        tensor(0.5000)
        """
        l1_loss = torch.abs(input - target)
        cond = l1_loss < self.beta
        loss = torch.where(
            cond, 0.5 * l1_loss**2 / self.beta, l1_loss - 0.5 * self.beta
        )
        if lds_weight is not None:
            loss *= lds_weight
        return torch.mean(loss)
