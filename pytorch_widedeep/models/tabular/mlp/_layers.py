from torch import nn

from pytorch_widedeep.wdtypes import *  # noqa: F403
from pytorch_widedeep.models._get_activation_fn import get_activation_fn


def dense_layer(
    inp: int,
    out: int,
    activation: str,
    p: float,
    bn: bool,
    linear_first: bool,
):
    # This is basically the LinBnDrop class at the fastai library
    if activation == "geglu":
        raise ValueError(
            "'geglu' activation is only used as 'transformer_activation' "
            "in transformer-based models"
        )
    act_fn = get_activation_fn(activation)
    layers = [nn.BatchNorm1d(out if linear_first else inp)] if bn else []
    if p != 0:
        layers.append(nn.Dropout(p))  # type: ignore[arg-type]
    lin = [nn.Linear(inp, out, bias=not bn), act_fn]
    layers = lin + layers if linear_first else layers + lin
    return nn.Sequential(*layers)


class MLP(nn.Module):
    def __init__(
        self,
        d_hidden: List[int],
        activation: str,
        dropout: Optional[Union[float, List[float]]],
        batchnorm: bool,
        batchnorm_last: bool,
        linear_first: bool,
    ):
        super(MLP, self).__init__()

        if not dropout:
            dropout = [0.0] * len(d_hidden)
        elif isinstance(dropout, float):
            dropout = [dropout] * len(d_hidden)

        self.mlp = nn.Sequential()
        for i in range(1, len(d_hidden)):
            self.mlp.add_module(
                "dense_layer_{}".format(i - 1),
                dense_layer(
                    d_hidden[i - 1],
                    d_hidden[i],
                    activation,
                    dropout[i - 1],
                    batchnorm and (i != len(d_hidden) - 1 or batchnorm_last),
                    linear_first,
                ),
            )

    def forward(self, X: Tensor) -> Tensor:
        return self.mlp(X)
