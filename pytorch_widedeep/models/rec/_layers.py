from typing import Dict, List, Tuple, Literal, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from pytorch_widedeep.models.tabular._base_tabular_model import (
    BaseTabularModelWithAttention,
)


class PseudoLinear(BaseTabularModelWithAttention):
    def __init__(
        self,
        column_idx: Dict[str, int],
        *,
        cat_embed_input: Optional[List[Tuple[str, int]]],
        cat_embed_dropout: Optional[float],
        use_cat_bias: Optional[bool],
        cat_embed_activation: Optional[str],
        continuous_cols: Optional[List[str]],
        cont_norm_layer: Optional[Literal["batchnorm", "layernorm"]],
        embed_continuous: Optional[bool],
        embed_continuous_method: Optional[Literal["piecewise", "periodic"]],
        cont_embed_dropout: Optional[float],
        cont_embed_activation: Optional[str],
        quantization_setup: Optional[Dict[str, List[float]]],
        n_frequencies: Optional[int],
        sigma: Optional[float],
        share_last_layer: Optional[bool],
        full_embed_dropout: Optional[bool],
    ):
        super(PseudoLinear, self).__init__(
            column_idx=column_idx,
            input_dim=1,
            cat_embed_input=cat_embed_input,
            cat_embed_dropout=cat_embed_dropout,
            use_cat_bias=use_cat_bias,
            cat_embed_activation=cat_embed_activation,
            shared_embed=None,
            add_shared_embed=None,
            frac_shared_embed=None,
            continuous_cols=continuous_cols,
            cont_norm_layer=cont_norm_layer,
            embed_continuous=embed_continuous,
            embed_continuous_method=embed_continuous_method,
            cont_embed_dropout=cont_embed_dropout,
            cont_embed_activation=cont_embed_activation,
            quantization_setup=quantization_setup,
            n_frequencies=n_frequencies,
            sigma=sigma,
            share_last_layer=share_last_layer,
            full_embed_dropout=full_embed_dropout,
        )

    def forward(self, X: Tensor) -> Tensor:
        return self._get_embeddings(X).sum(dim=1)


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

    def forward(self, query, user_behavior):
        # query: [batch_size, embedding_dim]
        # user_behavior: [batch_size, seq_len, embedding_dim]
        query = query.unsqueeze(1).expand(-1, user_behavior.size(1), -1)
        attn_input = torch.cat(
            [query, user_behavior, query - user_behavior, query * user_behavior], dim=-1
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
