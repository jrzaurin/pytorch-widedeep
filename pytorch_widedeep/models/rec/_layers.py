from typing import List, Literal, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class Dice(nn.Module):
    def __init__(self, input_dim: int):
        super(Dice, self).__init__()
        self.bn = nn.BatchNorm1d(input_dim, eps=1e-9)
        self.alpha = nn.Parameter(torch.zeros(input_dim))

    def forward(self, X: Tensor) -> Tensor:
        # assumes X has n_dim = 3
        x_p = self.bn(X.transpose(1, 2))
        p = torch.sigmoid(x_p)
        return X.mul(p) + self.alpha * X.mul(1 - p)


class ActivationUnit(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        activation: Literal["prelu", "dice"],
        proj_dim: Optional[int] = None,
    ):
        super(ActivationUnit, self).__init__()
        self.proj_dim = proj_dim if proj_dim is not None else embed_dim
        self.linear_in = nn.Linear(embed_dim * 4, self.proj_dim)
        if activation == "prelu":
            self.activation: nn.PReLU | Dice = nn.PReLU()
        elif activation == "dice":
            self.activation = Dice(self.proj_dim)
        self.linear_out = nn.Linear(self.proj_dim, 1)

    def forward(self, item: Tensor, user_behavior: Tensor) -> Tensor:
        # in my implementation:
        # item: [batch_size, 1, embedding_dim]
        # user_behavior: [batch_size, seq_len, embedding_dim]
        item = item.expand(-1, user_behavior.size(1), -1)
        attn_input = torch.cat(
            [item, user_behavior, item - user_behavior, item * user_behavior], dim=-1
        )
        attn_output = self.activation(self.linear_in(attn_input))
        attn_output = self.linear_out(attn_output).squeeze(-1)
        return F.softmax(attn_output, dim=1)


class CompressedInteractionNetwork(nn.Module):
    def __init__(self, num_cols: int, cin_layer_dims: List[int]):
        super(CompressedInteractionNetwork, self).__init__()
        self.num_cols = num_cols
        self.cin_layer_dims = cin_layer_dims
        self.cin_layers = nn.ModuleList()

        prev_layer_dim = num_cols
        for layer_dim in cin_layer_dims:
            self.cin_layers.append(
                nn.Conv1d(prev_layer_dim * num_cols, layer_dim, kernel_size=1)
            )
            prev_layer_dim = layer_dim

    def forward(self, X: Tensor) -> Tensor:
        batch_size, embed_dim = X.shape[0], X.shape[-1]
        prev_x = X
        cin_outs = []
        for cin_layer in self.cin_layers:
            x_i = torch.einsum("b m d, b h d  -> b m h d", X, prev_x)
            x_i = x_i.reshape(batch_size, self.num_cols * prev_x.shape[1], embed_dim)
            x_i = cin_layer(x_i)
            cin_outs.append(x_i.sum(2))
            prev_x = x_i

        return torch.cat(cin_outs, dim=1)
