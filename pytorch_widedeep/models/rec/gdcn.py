from typing import Dict, List, Tuple, Union, Literal, Optional

import torch
from torch import nn

from pytorch_widedeep.models.tabular.mlp._layers import MLP
from pytorch_widedeep.models.tabular._base_tabular_model import (
    BaseTabularModelWithoutAttention,
)


class GatedCrossLayer(nn.Module):
    """
    Single Gated Cross Layer implementing:
    c_(i+1) = c_0 ⊙ (W^c × c_i + b) ⊙ σ(W^g × c_i) + c_i
    """

    def __init__(self, input_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.cross_weight = nn.Parameter(torch.empty(input_dim, input_dim))
        self.gate_weight = nn.Parameter(torch.empty(input_dim, input_dim))
        self.bias = nn.Parameter(torch.zeros(input_dim))
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize layer parameters"""
        nn.init.xavier_uniform_(self.cross_weight)
        nn.init.xavier_uniform_(self.gate_weight)

    def forward(self, x0: torch.Tensor, xi: torch.Tensor) -> torch.Tensor:
        cross_term = torch.einsum("bi,ij->bj", xi, self.cross_weight)
        cross_term = cross_term + self.bias
        cross_term = x0 * cross_term
        gate_term = torch.einsum("bi,ij->bj", xi, self.gate_weight)
        gate_term = torch.sigmoid(gate_term)
        return cross_term * gate_term + xi


class GatedCrossNetwork(nn.Module):
    def __init__(self, input_dim: int, num_layers: int):
        super().__init__()
        self.num_layers = num_layers
        self.input_dim = input_dim
        self.layers = nn.ModuleList(
            [GatedCrossLayer(input_dim) for _ in range(num_layers)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = x
        xi = x

        for layer in self.layers:
            xi = layer(x0, xi)

        return xi


class GatedCrossDeepNetwork(BaseTabularModelWithoutAttention):
    def __init__(
        self,
        column_idx: Dict[str, int],
        *,
        num_cross_layers: int = 3,
        structure: Literal["stacked", "parallel"] = "parallel",
        cat_embed_input: Optional[List[Tuple[str, int, int]]] = None,
        cat_embed_dropout: Optional[float] = None,
        use_cat_bias: Optional[bool] = None,
        cat_embed_activation: Optional[str] = None,
        continuous_cols: Optional[List[str]] = None,
        cont_norm_layer: Optional[Literal["batchnorm", "layernorm"]] = None,
        embed_continuous: Optional[bool] = None,
        embed_continuous_method: Optional[
            Literal["standard", "piecewise", "periodic"]
        ] = None,
        cont_embed_dim: Optional[int] = None,
        cont_embed_dropout: Optional[float] = None,
        cont_embed_activation: Optional[str] = None,
        quantization_setup: Optional[Dict[str, List[float]]] = None,
        n_frequencies: Optional[int] = None,
        sigma: Optional[float] = None,
        share_last_layer: Optional[bool] = None,
        full_embed_dropout: Optional[bool] = None,
        mlp_hidden_dims: List[int] = [200, 100],
        mlp_activation: str = "relu",
        mlp_dropout: Union[float, List[float]] = 0.1,
        mlp_batchnorm: bool = False,
        mlp_batchnorm_last: bool = False,
        mlp_linear_first: bool = True,
    ):
        super(GatedCrossDeepNetwork, self).__init__(
            column_idx=column_idx,
            cat_embed_input=cat_embed_input,
            cat_embed_dropout=cat_embed_dropout,
            use_cat_bias=use_cat_bias,
            cat_embed_activation=cat_embed_activation,
            continuous_cols=continuous_cols,
            cont_norm_layer=cont_norm_layer,
            embed_continuous=embed_continuous,
            embed_continuous_method=embed_continuous_method,
            cont_embed_dim=cont_embed_dim,
            cont_embed_dropout=cont_embed_dropout,
            cont_embed_activation=cont_embed_activation,
            quantization_setup=quantization_setup,
            n_frequencies=n_frequencies,
            sigma=sigma,
            share_last_layer=share_last_layer,
            full_embed_dropout=full_embed_dropout,
        )

        self.num_cross_layers = num_cross_layers
        self.structure = structure

        embeddings_output_dim = self.cat_out_dim + self.cont_out_dim
        mlp_hidden_dims = [embeddings_output_dim] + mlp_hidden_dims
        self.deep_network = MLP(
            mlp_hidden_dims,
            mlp_activation,
            mlp_dropout,
            mlp_batchnorm,
            mlp_batchnorm_last,
            mlp_linear_first,
        )

        self.cross_network = GatedCrossNetwork(embeddings_output_dim, num_cross_layers)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        x = self._get_embeddings(X)
        cross_output = self.cross_network(x)

        if self.structure == "stacked":
            deep_output = self.deep_network(cross_output)
            return deep_output
        else:  # parallel
            deep_output = self.deep_network(x)
            return torch.cat([cross_output, deep_output], dim=1)

    @property
    def output_dim(self) -> int:
        if self.structure == "stacked":
            return self.mlp_hidden_dims[-1]
        else:  # parallel
            return self.mlp_hidden_dims[-1] + self.cat_out_dim + self.cont_out_dim
