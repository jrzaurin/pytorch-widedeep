from torch import nn

from ..wdtypes import *


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
    >>> X = torch.empty(4, 4).random_(2)
    >>> wide = Wide(wide_dim=X.size(0), pred_dim=1)
    >>> wide(X)
    tensor([[-0.8841],
            [-0.8633],
            [-1.2713],
            [-0.4762]], grad_fn=<AddmmBackward>)
    """

    def __init__(self, wide_dim: int, pred_dim: int = 1):
        super(Wide, self).__init__()
        self.wide_linear = nn.Linear(wide_dim, pred_dim)

    def forward(self, X: Tensor) -> Tensor:  # type: ignore
        r"""Forward pass. Simply connecting the one-hot encoded input with the
        ouput neuron(s) """
        out = self.wide_linear(X.float())
        return out
