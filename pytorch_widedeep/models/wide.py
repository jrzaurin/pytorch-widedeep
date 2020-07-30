from torch import nn

from ..wdtypes import *

import torch
import math


class Wide(nn.Module):
    r"""Simple linear layer that will receive the one-hot encoded `'wide'`
    input and connect it to the output neuron(s).

    Parameters
    -----------
    wide_dim: int
        size of the input tensor
    pred_dim: int
        size of the ouput tensor containing the predictions

    Attributes
    -----------
    wide_linear: :obj:`nn.Module`
        the linear layer that comprises the wide branch of the model

    Examples
    --------
    >>> import torch
    >>> from pytorch_widedeep.models import Wide
    >>> X = torch.empty(4, 5).random_(2)
    >>> wide = Wide(wide_dim=X.size(1), pred_dim=1)
    >>> wide(X)
    tensor([[-0.8841],
            [-0.8633],
            [-1.2713],
            [-0.4762]], grad_fn=<AddmmBackward>)
    """

    def __init__(self, vocab_size: int, pred_dim: int = 1):
        super(Wide, self).__init__()
        # self.wide_linear = nn.Linear(wide_dim, pred_dim)
        self.wide_linear = nn.Embedding(
            vocab_size + 1, pred_dim, padding_idx=0)
        # Sum(Embedding) + bias = OneHotVector + Linear
        self.bias = nn.Parameter(torch.Tensor(pred_dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # initialize Embedding and bias like nn.Linear
        # Reference: https://pytorch.org/docs/master/_modules/torch/nn/modules/linear.html#Linear
        nn.init.kaiming_uniform_(self.wide_linear.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.wide_linear.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, X: Tensor) -> Tensor:  # type: ignore
        r"""Forward pass.

        Parameters
        -----------
        wide_dim: Tensor
            idx of the feature. all type of categories in a dictionary. start from 1. 0 is for padding.
        """
        # out = self.wide_linear(X.float())
        out = self.wide_linear(X).sum(dim=1) + self.bias
        return out
