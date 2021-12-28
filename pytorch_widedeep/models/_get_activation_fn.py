import torch.nn.functional as F
from torch import nn

allowed_activations = [
    "relu",
    "leaky_relu",
    "tanh",
    "gelu",
    "geglu",
    "reglu",
]


class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


class REGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


def get_activation_fn(activation):
    if activation == "relu":
        return nn.ReLU(inplace=True)
    elif activation == "leaky_relu":
        return nn.LeakyReLU(inplace=True)
    elif activation == "tanh":
        return nn.Tanh()
    elif activation == "gelu":
        return nn.GELU()
    elif activation == "geglu":
        return GEGLU()
    elif activation == "reglu":
        return REGLU()
    elif activation == "softplus":
        return nn.Softplus()
    else:
        raise ValueError(
            "Only the following activation functions are currently "
            "supported: {}. Note that 'geglu' and 'reglu' "
            "should only be used as transformer's activations".format(
                ", ".join(allowed_activations)
            )
        )
