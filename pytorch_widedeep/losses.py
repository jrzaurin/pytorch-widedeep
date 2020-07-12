import torch
import torch.nn as nn
import torch.nn.functional as F

from .wdtypes import *

use_cuda = torch.cuda.is_available()


class FocalLoss(nn.Module):
    r"""Implementation of the `focal loss
    <https://arxiv.org/pdf/1708.02002.pdf>`_ for both binary and
    multiclass classification

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

    def forward(self, input: Tensor, target: Tensor) -> Tensor:  # type: ignore
        r"""Focal Loss computation

        Parameters
        ----------
        input: Tensor
            input tensor with predictions (not probabilities)
        target: Tensor
            target tensor with the actual classes

        Examples
        --------
        >>> import torch
        >>> import numpy as np
        >>> from pytorch_widedeep.losses import FocalLoss
        >>>
        >>> # BINARY
        >>> target = torch.from_numpy(np.random.choice(2, 10))
        >>> input = torch.rand(10,1)
        >>> FocalLoss()(input, target)
        tensor(0.1604)
        >>>
        >>> # MULTICLASS
        >>> target = torch.from_numpy(np.random.choice(3, 10))
        >>> input = torch.rand(10,3)
        >>> FocalLoss()(input, target)
        tensor(0.3024)
        """
        input_prob = torch.sigmoid(input)
        if input.size(1) == 1:
            input_prob = torch.cat([1 - input_prob, input_prob], axis=1)  # type: ignore
            num_class = 2
        else:
            num_class = input_prob.size(1)
        binary_target = torch.eye(num_class)[target.long()]
        if use_cuda:
            binary_target = binary_target.cuda()
        binary_target = binary_target.contiguous()
        weight = self._get_weight(input_prob, binary_target)
        return F.binary_cross_entropy(
            input_prob, binary_target, weight, reduction="mean"
        )
