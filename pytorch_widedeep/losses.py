import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_widedeep.wdtypes import (
    List,
    Tuple,
    Union,
    Tensor,
    Literal,
    Optional,
)

use_cuda = torch.cuda.is_available()


class MSELoss(nn.Module):
    r"""Mean square error loss with the option of using Label Smooth
    Distribution (LDS)

    LDS is based on
    [Delving into Deep Imbalanced Regression](https://arxiv.org/abs/2102.09554).
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
            Tensor of weights that will multiply the loss value.

        Examples
        --------
        >>> import torch
        >>> from pytorch_widedeep.losses import MSELoss
        >>>
        >>> target = torch.tensor([1, 1.2, 0, 2]).view(-1, 1)
        >>> input = torch.tensor([0.6, 0.7, 0.3, 0.8]).view(-1, 1)
        >>> lds_weight = torch.tensor([0.1, 0.2, 0.3, 0.4]).view(-1, 1)
        >>> loss = MSELoss()(input, target, lds_weight)
        """
        loss = (input - target) ** 2
        if lds_weight is not None:
            loss *= lds_weight
        return torch.mean(loss)


class MSLELoss(nn.Module):
    r"""Mean square log error loss with the option of using Label Smooth
    Distribution (LDS)

    LDS is based on
    [Delving into Deep Imbalanced Regression](https://arxiv.org/abs/2102.09554).
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
            Tensor of weights that will multiply the loss value.

        Examples
        --------
        >>> import torch
        >>> from pytorch_widedeep.losses import MSLELoss
        >>>
        >>> target = torch.tensor([1, 1.2, 0, 2]).view(-1, 1)
        >>> input = torch.tensor([0.6, 0.7, 0.3, 0.8]).view(-1, 1)
        >>> lds_weight = torch.tensor([0.1, 0.2, 0.3, 0.4]).view(-1, 1)
        >>> loss = MSLELoss()(input, target, lds_weight)
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

    LDS is based on
    [Delving into Deep Imbalanced Regression](https://arxiv.org/abs/2102.09554).
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
            Tensor of weights that will multiply the loss value.

        Examples
        --------
        >>> import torch
        >>> from pytorch_widedeep.losses import RMSELoss
        >>>
        >>> target = torch.tensor([1, 1.2, 0, 2]).view(-1, 1)
        >>> input = torch.tensor([0.6, 0.7, 0.3, 0.8]).view(-1, 1)
        >>> lds_weight = torch.tensor([0.1, 0.2, 0.3, 0.4]).view(-1, 1)
        >>> loss = RMSELoss()(input, target, lds_weight)
        """
        loss = (input - target) ** 2
        if lds_weight is not None:
            loss *= lds_weight
        return torch.sqrt(torch.mean(loss))


class RMSLELoss(nn.Module):
    r"""Root mean square log error loss adjusted for the possibility of using Label
    Smooth Distribution (LDS)

    LDS is based on
    [Delving into Deep Imbalanced Regression](https://arxiv.org/abs/2102.09554).
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
            Tensor of weights that will multiply the loss value.

        Examples
        --------
        >>> import torch
        >>> from pytorch_widedeep.losses import RMSLELoss
        >>>
        >>> target = torch.tensor([1, 1.2, 0, 2]).view(-1, 1)
        >>> input = torch.tensor([0.6, 0.7, 0.3, 0.8]).view(-1, 1)
        >>> lds_weight = torch.tensor([0.1, 0.2, 0.3, 0.4]).view(-1, 1)
        >>> loss = RMSLELoss()(input, target, lds_weight)
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

    $$
    Loss = max(q \times (y-y_{pred}), (1-q) \times (y_{pred}-y))
    $$

    All credits go to the implementation at
    [pytorch-forecasting](https://pytorch-forecasting.readthedocs.io/en/latest/_modules/pytorch_forecasting/metrics.html#QuantileLoss).

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
        >>> loss = qloss(input, target)
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
    r"""Implementation of the [Focal loss](https://arxiv.org/pdf/1708.02002.pdf)
    for both binary and multiclass classification:

    $$
    FL(p_t) = \alpha (1 - p_t)^{\gamma} log(p_t)
    $$

    where, for a case of a binary classification problem

    $$
    \begin{equation} p_t= \begin{cases}p, & \text{if $y=1$}.\\1-p, & \text{otherwise}. \end{cases} \end{equation}
    $$

    Parameters
    ----------
    alpha: float
        Focal Loss `alpha` parameter
    gamma: float
        Focal Loss `gamma` parameter
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
        >>> loss = FocalLoss()(input, target)
        >>>
        >>> # MULTICLASS
        >>> target = torch.tensor([1, 0, 2]).view(-1, 1)
        >>> input = torch.tensor([[0.2, 0.5, 0.3], [0.8, 0.1, 0.1], [0.7, 0.2, 0.1]])
        >>> loss = FocalLoss()(input, target)
        """
        input_prob = torch.sigmoid(input)
        if input.size(1) == 1:
            input_prob = torch.cat([1 - input_prob, input_prob], axis=1)  # type: ignore
            num_class = 2
        else:
            num_class = input_prob.size(1)
        binary_target = torch.eye(num_class)[target.squeeze().cpu().long()]
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
    `BayesianSELoss` and the latter does not need a scale/noise_tolerance
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
    the original publication
    [Weight Uncertainty in Neural Networks](https://arxiv.org/abs/1505.05424).
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
        >>> loss = BayesianSELoss()(input, target)
        """
        return (0.5 * (input - target) ** 2).sum()


class TweedieLoss(nn.Module):
    """
    Tweedie loss for extremely unbalanced zero-inflated data

    All credits go to Wenbo Shi. See
    [this post](https://towardsdatascience.com/tweedie-loss-function-for-right-skewed-data-2c5ca470678f)
    and the [original publication](https://arxiv.org/abs/1811.10192) for details.
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
        >>> loss = TweedieLoss()(input, target, lds_weight)
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

    See [A Deep Probabilistic Model for Customer Lifetime Value Prediction](https://arxiv.org/pdf/1912.07753.pdf)
    and the corresponding
    [code](https://github.com/google/lifetime_value/blob/master/lifetime_value/zero_inflated_lognormal.py).
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
        >>> loss = ZILNLoss()(input, target)
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

    LDS is based on
    [Delving into Deep Imbalanced Regression](https://arxiv.org/abs/2102.09554).
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
        >>> loss = L1Loss()(input, target)
        """
        loss = F.l1_loss(input, target, reduction="none")
        if lds_weight is not None:
            loss *= lds_weight
        return torch.mean(loss)


class FocalR_L1Loss(nn.Module):
    r"""Focal-R L1 loss

    Based on [Delving into Deep Imbalanced Regression](https://arxiv.org/abs/2102.09554).

    Parameters
    ----------
    beta: float
        Focal Loss `beta` parameter in their implementation
    gamma: float
        Focal Loss `gamma` parameter
    activation_fn: str, default = "sigmoid"
        Activation function to be used during the computation of the loss.
        Possible values are _'sigmoid'_ and _'tanh'_. See the original
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
        >>> loss = FocalR_L1Loss()(input, target)
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

    Based on [Delving into Deep Imbalanced Regression](https://arxiv.org/abs/2102.09554).

    Parameters
    ----------
    beta: float
        Focal Loss `beta` parameter in their implementation
    gamma: float
        Focal Loss `gamma` parameter
    activation_fn: str, default = "sigmoid"
        Activation function to be used during the computation of the loss.
        Possible values are _'sigmoid'_ and _'tanh'_. See the original
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
        >>> loss = FocalR_MSELoss()(input, target)
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

    Based on [Delving into Deep Imbalanced Regression](https://arxiv.org/abs/2102.09554).

    Parameters
    ----------
    beta: float
        Focal Loss `beta` parameter in their implementation
    gamma: float
        Focal Loss `gamma` parameter
    activation_fn: str, default = "sigmoid"
        Activation function to be used during the computation of the loss.
        Possible values are _'sigmoid'_ and _'tanh'_. See the original
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
        >>> loss = FocalR_RMSELoss()(input, target)
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

    Based on [Delving into Deep Imbalanced Regression](https://arxiv.org/abs/2102.09554).
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

        Examples
        --------
        >>> import torch
        >>>
        >>> from pytorch_widedeep.losses import HuberLoss
        >>>
        >>> target = torch.tensor([1, 1.2, 0, 2]).view(-1, 1)
        >>> input = torch.tensor([0.6, 0.7, 0.3, 0.8]).view(-1, 1)
        >>> loss = HuberLoss()(input, target)
        """
        l1_loss = torch.abs(input - target)
        cond = l1_loss < self.beta
        loss = torch.where(
            cond, 0.5 * l1_loss**2 / self.beta, l1_loss - 0.5 * self.beta
        )
        if lds_weight is not None:
            loss *= lds_weight
        return torch.mean(loss)


class InfoNCELoss(nn.Module):
    r"""InfoNCE Loss. Loss applied during the Contrastive Denoising Self
    Supervised Pre-training routine available in this library

    :information_source: **NOTE**: This loss is in principle not exposed to
     the user, as it is used internally in the library, but it is included
     here for completion.

    See [SAINT: Improved Neural Networks for Tabular Data via Row Attention
    and Contrastive Pre-Training](https://arxiv.org/abs/2106.01342) and
    references therein

    Partially inspired by the code in this [repo](https://github.com/RElbers/info-nce-pytorch)

    Parameters
    ----------
    temperature: float, default = 0.1
        The logits are divided by the temperature before computing the loss value
    reduction: str, default = "mean"
        Loss reduction method
    """

    def __init__(self, temperature: float = 0.1, reduction: str = "mean"):
        super(InfoNCELoss, self).__init__()

        self.temperature = temperature
        self.reduction = reduction

    def forward(self, g_projs: Tuple[Tensor, Tensor]) -> Tensor:
        r"""
        Parameters
        ----------
        g_projs: Tuple
            Tuple with the two tensors corresponding to the output of the two
            projection heads, as described 'SAINT: Improved Neural Networks
            for Tabular Data via Row Attention and Contrastive Pre-Training'.

        Examples
        --------
        >>> import torch
        >>> from pytorch_widedeep.losses import InfoNCELoss
        >>> g_projs = (torch.rand(3, 5, 16), torch.rand(3, 5, 16))
        >>> loss = InfoNCELoss()
        >>> res = loss(g_projs)
        """
        z, z_ = g_projs[0], g_projs[1]

        norm_z = F.normalize(z, dim=-1).flatten(1)
        norm_z_ = F.normalize(z_, dim=-1).flatten(1)

        logits = (norm_z @ norm_z_.t()) / self.temperature
        logits_ = (norm_z_ @ norm_z.t()) / self.temperature

        # the target/labels are the entries on the diagonal
        target = torch.arange(len(norm_z), device=norm_z.device)

        loss = F.cross_entropy(logits, target, reduction=self.reduction)
        loss_ = F.cross_entropy(logits_, target, reduction=self.reduction)

        return (loss + loss_) / 2.0


class DenoisingLoss(nn.Module):
    r"""Denoising Loss. Loss applied during the Contrastive Denoising Self
    Supervised Pre-training routine available in this library

    :information_source: **NOTE**: This loss is in principle not exposed to
     the user, as it is used internally in the library, but it is included
     here for completion.

    See [SAINT: Improved Neural Networks for Tabular Data via Row Attention
    and Contrastive Pre-Training](https://arxiv.org/abs/2106.01342) and
    references therein

    Parameters
    ----------
    lambda_cat: float, default = 1.
        Multiplicative factor that will be applied to loss associated to the
        categorical features
    lambda_cont: float, default = 1.
        Multiplicative factor that will be applied to loss associated to the
        continuous features
    reduction: str, default = "mean"
        Loss reduction method
    """

    def __init__(
        self, lambda_cat: float = 1.0, lambda_cont: float = 1.0, reduction: str = "mean"
    ):
        super(DenoisingLoss, self).__init__()

        self.lambda_cat = lambda_cat
        self.lambda_cont = lambda_cont
        self.reduction = reduction

    def forward(
        self,
        x_cat_and_cat_: Optional[
            Union[List[Tuple[Tensor, Tensor]], Tuple[Tensor, Tensor]]
        ],
        x_cont_and_cont_: Optional[
            Union[List[Tuple[Tensor, Tensor]], Tuple[Tensor, Tensor]]
        ],
    ) -> Tensor:
        r"""
        Parameters
        ----------
        x_cat_and_cat_: tuple of Tensors or lists of tuples
            Tuple of tensors containing the raw input features and their
            encodings, referred in the SAINT paper as $x$ and $x''$
            respectively. If one denoising MLP is used per categorical
            feature `x_cat_and_cat_` will be a list of tuples, one per
            categorical feature
        x_cont_and_cont_: tuple of Tensors or lists of tuples
            same as `x_cat_and_cat_` but for continuous columns

        Examples
        --------
        >>> import torch
        >>> from pytorch_widedeep.losses import DenoisingLoss
        >>> x_cat_and_cat_ = (torch.empty(3).random_(3).long(), torch.randn(3, 3))
        >>> x_cont_and_cont_ = (torch.randn(3, 1), torch.randn(3, 1))
        >>> loss = DenoisingLoss()
        >>> res = loss(x_cat_and_cat_, x_cont_and_cont_)
        """

        loss_cat = (
            self._compute_cat_loss(x_cat_and_cat_)
            if x_cat_and_cat_ is not None
            else torch.tensor(0.0)
        )
        loss_cont = (
            self._compute_cont_loss(x_cont_and_cont_)
            if x_cont_and_cont_ is not None
            else torch.tensor(0.0)
        )

        return self.lambda_cat * loss_cat + self.lambda_cont * loss_cont

    def _compute_cat_loss(
        self, x_cat_and_cat_: Union[List[Tuple[Tensor, Tensor]], Tuple[Tensor, Tensor]]
    ) -> Tensor:
        loss_cat = torch.tensor(0.0, device=self._get_device(x_cat_and_cat_))
        if isinstance(x_cat_and_cat_, list):
            for x, x_ in x_cat_and_cat_:
                loss_cat += F.cross_entropy(x_, x, reduction=self.reduction)
        elif isinstance(x_cat_and_cat_, tuple):
            x, x_ = x_cat_and_cat_
            loss_cat += F.cross_entropy(x_, x, reduction=self.reduction)

        return loss_cat

    def _compute_cont_loss(self, x_cont_and_cont_) -> Tensor:
        loss_cont = torch.tensor(0.0, device=self._get_device(x_cont_and_cont_))
        if isinstance(x_cont_and_cont_, list):
            for x, x_ in x_cont_and_cont_:
                loss_cont += F.mse_loss(x_, x, reduction=self.reduction)
        elif isinstance(x_cont_and_cont_, tuple):
            x, x_ = x_cont_and_cont_
            loss_cont += F.mse_loss(x_, x, reduction=self.reduction)

        return loss_cont

    @staticmethod
    def _get_device(
        x_and_x_: Union[List[Tuple[Tensor, Tensor]], Tuple[Tensor, Tensor]]
    ):
        if isinstance(x_and_x_, tuple):
            device = x_and_x_[0].device
        elif isinstance(x_and_x_, list):
            device = x_and_x_[0][0].device
        return device


class EncoderDecoderLoss(nn.Module):
    r"""'_Standard_' Encoder Decoder Loss. Loss applied during the Endoder-Decoder
     Self-Supervised Pre-Training routine available in this library

    :information_source: **NOTE**: This loss is in principle not exposed to
     the user, as it is used internally in the library, but it is included
     here for completion.

    The implementation of this lost is based on that at the
    [tabnet repo](https://github.com/dreamquark-ai/tabnet), which is in itself an
    adaptation of that in the original paper [TabNet: Attentive
    Interpretable Tabular Learning](https://arxiv.org/abs/1908.07442).

    Parameters
    ----------
    eps: float
        Simply a small number to avoid dividing by zero
    """

    def __init__(self, eps: float = 1e-9):
        super(EncoderDecoderLoss, self).__init__()
        self.eps = eps

    def forward(self, x_true: Tensor, x_pred: Tensor, mask: Tensor) -> Tensor:
        r"""
        Parameters
        ----------
        x_true: Tensor
            Embeddings of the input data
        x_pred: Tensor
            Reconstructed embeddings
        mask: Tensor
            Mask with 1s indicated that the reconstruction, and therefore the
            loss, is based on those features.

        Examples
        --------
        >>> import torch
        >>> from pytorch_widedeep.losses import EncoderDecoderLoss
        >>> x_true = torch.rand(3, 3)
        >>> x_pred = torch.rand(3, 3)
        >>> mask = torch.empty(3, 3).random_(2)
        >>> loss = EncoderDecoderLoss()
        >>> res = loss(x_true, x_pred, mask)
        """

        errors = x_pred - x_true

        reconstruction_errors = torch.mul(errors, mask) ** 2

        x_true_means = torch.mean(x_true, dim=0)
        x_true_means[x_true_means == 0] = 1

        x_true_stds = torch.std(x_true, dim=0) ** 2
        x_true_stds[x_true_stds == 0] = x_true_means[x_true_stds == 0]

        features_loss = torch.matmul(reconstruction_errors, 1 / x_true_stds)
        nb_reconstructed_variables = torch.sum(mask, dim=1)
        features_loss_norm = features_loss / (nb_reconstructed_variables + self.eps)

        loss = torch.mean(features_loss_norm)

        return loss
