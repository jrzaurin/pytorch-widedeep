import math

import torch
from torch import nn

from pytorch_widedeep.wdtypes import *  # noqa: F403


class Wide(nn.Module):
    r"""wide (linear) component

    Linear model implemented via an Embedding layer connected to the output
    neuron(s).

    Parameters
    -----------
    wide_dim: int
        size of the Embedding layer. `wide_dim` is the summation of all the
        individual values for all the features that go through the wide
        component. For example, if the wide component receives 2 features with
        5 individual values each, `wide_dim = 10`
    pred_dim: int, default = 1
        size of the ouput tensor containing the predictions

    Attributes
    -----------
    wide_linear: ``nn.Module``
        the linear layer that comprises the wide branch of the model

    Examples
    --------
    >>> import torch
    >>> from pytorch_widedeep.models import Wide
    >>> X = torch.empty(4, 4).random_(6)
    >>> wide = Wide(wide_dim=X.unique().size(0), pred_dim=1)
    >>> out = wide(X)
    """

    def __init__(self, wide_dim: int, pred_dim: int = 1):
        super(Wide, self).__init__()
        # Embeddings: val + 1 because 0 is reserved for padding/unseen cateogories.
        self.wide_linear = nn.Embedding(wide_dim + 1, pred_dim, padding_idx=0)
        # (Sum(Embedding) + bias) is equivalent to (OneHotVector + Linear)
        self.bias = nn.Parameter(torch.zeros(pred_dim))
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        r"""initialize Embedding and bias like nn.Linear. See `original
        implementation
        <https://pytorch.org/docs/stable/_modules/torch/nn/modules/linear.html#Linear>`_.
        """
        nn.init.kaiming_uniform_(self.wide_linear.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.wide_linear.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, X: Tensor) -> Tensor:  # type: ignore
        r"""Forward pass. Simply connecting the Embedding layer with the ouput
        neuron(s)"""
        out = self.wide_linear(X.long()).sum(dim=1) + self.bias
        return out
