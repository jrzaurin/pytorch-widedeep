import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_widedeep.wdtypes import *  # noqa: F403

use_cuda = torch.cuda.is_available()


class TweedieLoss(nn.Module):
    """
    Tweedie loss for extremely unbalanced zero-inflated data``
    All credits go to `Wenbo Shi
    <https://towardsdatascience.com/tweedie-loss-function-for-right-skewed-data-2c5ca470678f> and
    <https://arxiv.org/abs/1811.10192>`
    """

    def __init__(self):
        super().__init__()

    def forward(self, input: Tensor, target: Tensor, p=1.5) -> Tensor:
        assert (
            input.min() > 0
        ), """All input values must be >=0, if your model is predicting
            values <0 try to enforce positive values by activation function
            on last layer with `trainer.enforce_positive_output=True`"""
        assert target.min() >= 0, "All target values must be >=0"
        loss = -target * torch.pow(input, 1 - p) / (1 - p) + torch.pow(input, 2 - p) / (
            2 - p
        )
        return torch.mean(loss)


class QuantileLoss(nn.Module):
    """
    Quantile loss, i.e. a quantile of ``q=0.5`` will give half of the mean absolute error as it is calcualted as

    Defined as ``max(q * (y-y_pred), (1-q) * (y_pred-y))``
    All credits go to `pytorch-forecasting
    <https://pytorch-forecasting.readthedocs.io/en/latest/_modules/pytorch_forecasting/metrics.html#QuantileLoss>`
    """

    def __init__(
        self,
        quantiles: List[float] = [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98],
    ):
        """
        Quantile loss

        Args:
            quantiles: quantiles for metric
        """
        super().__init__()
        self.quantiles = quantiles

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        assert input.shape == torch.Size([target.shape[0], len(self.quantiles)]), (
            f"Wrong shape of input, pred_dim of the model that is using QuantileLoss must be equal "
            f"to number of quantiles, i.e. {len(self.quantiles)}."
        )
        target = target.view(-1, 1).float()
        losses = []
        for i, q in enumerate(self.quantiles):
            errors = target - input[..., i]
            losses.append(torch.max((q - 1) * errors, q * errors).unsqueeze(-1))
        loss = torch.cat(losses, dim=2)

        return torch.mean(loss)


class ZILNLoss(nn.Module):
    r"""Adjusted implementation of the `Zero Inflated LogNormal loss
    <https://arxiv.org/pdf/1912.07753.pdf>` and its `code
    <https://github.com/google/lifetime_value/blob/master/lifetime_value/zero_inflated_lognormal.py>`
    """

    def __init__(self):
        super().__init__()

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        r"""
        Parameters
        ----------
        input: Tensor
            input tensor with predictions (not probabilities) with spape (N,3), where N is the batch size
        target: Tensor
            target tensor with the actual classes

        Examples
        --------
        >>> import torch
        >>>
        >>> from pytorch_widedeep.losses import ZILNLoss
        >>>
        >>> # REGRESSION
        >>> target = torch.tensor([[0., 1.5]]).view(-1, 1)
        >>> input = torch.tensor([[.1, .2, .3], [.4, .5, .6]])
        >>> ZILNLoss()(input, target)
        tensor(1.3114)
        """
        positive = target > 0
        positive = positive.float()

        assert input.shape == torch.Size(
            [target.shape[0], 3]
        ), "Wrong shape of input, pred_dim of the model that is using ZILNLoss must be equal to 3."
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
            input tensor with predictions (not probabilities)
        target: Tensor
            target tensor with the actual classes

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


class MSLELoss(nn.Module):
    r"""mean squared log error"""

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        r"""
        Parameters
        ----------
        input: Tensor
            input tensor with predictions (not probabilities)
        target: Tensor
            target tensor with the actual classes

        Examples
        --------
        >>> import torch
        >>> from pytorch_widedeep.losses import MSLELoss
        >>>
        >>> target = torch.tensor([1, 1.2, 0, 2]).view(-1, 1)
        >>> input = torch.tensor([0.6, 0.7, 0.3, 0.8]).view(-1, 1)
        >>> MSLELoss()(input, target)
        tensor(0.1115)
        """
        assert (
            input.min() >= 0
        ), """All input values must be >=0, if your model is predicting
            values <0 try to enforce positive values by activation function
            on last layer with `trainer.enforce_positive_output=True`"""
        assert target.min() >= 0, "All target values must be >=0"
        return self.mse(torch.log(input + 1), torch.log(target + 1))


class RMSELoss(nn.Module):
    r"""root mean squared error"""

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        r"""
        Parameters
        ----------
        input: Tensor
            input tensor with predictions (not probabilities)
        target: Tensor
            target tensor with the actual classes

        Examples
        --------
        >>> import torch
        >>> from pytorch_widedeep.losses import RMSELoss
        >>>
        >>> target = torch.tensor([1, 1.2, 0, 2]).view(-1, 1)
        >>> input = torch.tensor([0.6, 0.7, 0.3, 0.8]).view(-1, 1)
        >>> RMSELoss()(input, target)
        tensor(0.6964)
        """
        return torch.sqrt(self.mse(input, target))


class RMSLELoss(nn.Module):
    r"""root mean squared log error"""

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        r"""
        Parameters
        ----------
        input: Tensor
            input tensor with predictions (not probabilities)
        target: Tensor
            target tensor with the actual classes

        Examples
        --------
        >>> import torch
        >>> from pytorch_widedeep.losses import RMSLELoss
        >>>
        >>> target = torch.tensor([1, 1.2, 0, 2]).view(-1, 1)
        >>> input = torch.tensor([0.6, 0.7, 0.3, 0.8]).view(-1, 1)
        >>> RMSLELoss()(input, target)
        tensor(0.3339)
        """
        assert (
            input.min() >= 0
        ), """All input values must be >=0, if your model is predicting
            values <0 try to enforce positive values by activation function
            on last layer with `trainer.enforce_positive_output=True`"""
        assert target.min() >= 0, "All target values must be >=0"
        return torch.sqrt(self.mse(torch.log(input + 1), torch.log(target + 1)))
