import torch.nn.functional as F
from torch import nn

allowed_activations = [
    "relu",
    "leaky_relu",
    "tanh",
    "gelu",
    "geglu",
    "reglu",
    "softplus",
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
    if activation == "leaky_relu":
        return nn.LeakyReLU(inplace=True)
    if activation == "tanh":
        return nn.Tanh()
    if activation == "gelu":
        return nn.GELU()
    if activation == "geglu":
        return GEGLU()
    if activation == "reglu":
        return REGLU()
    if activation == "softplus":
        return nn.Softplus()
