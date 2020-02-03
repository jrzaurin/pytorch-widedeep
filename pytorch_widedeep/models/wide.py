from torch import nn
from ..wdtypes import *


class Wide(nn.Module):
    r"""simple linear layer between the one-hot encoded wide input and the output
    neuron.

    Parameters
    ----------
    wide_dim: Int
        size of the input tensor
    output_dim: Int
        size of the ouput tensor

    Attributes
    ----------
    wide_linear: nn.Module
        the linear layer that comprises the wide branch of the model

    Example
    --------
    >>> import torch
    >>> from pytorch_widedeep.models import Wide
    >>> X = torch.empty(4, 4).random_(2)
    >>> wide = Wide(wide_dim=X.size(0), output_dim=1)
    >>> wide(X)
    tensor([[-0.8841],
            [-0.8633],
            [-1.2713],
            [-0.4762]], grad_fn=<AddmmBackward>)
    """

    def __init__(self, wide_dim: int, output_dim: int = 1):
        super(Wide, self).__init__()
        self.wide_linear = nn.Linear(wide_dim, output_dim)

    def forward(self, X: Tensor) -> Tensor:  # type: ignore
        out = self.wide_linear(X.float())
        return out
