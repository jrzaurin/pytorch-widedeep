from typing import Dict, List, Tuple, Union, Literal, Optional

import torch
from torch import nn

from pytorch_widedeep.models.tabular.mlp._layers import MLP
from pytorch_widedeep.models.tabular._base_tabular_model import (
    BaseTabularModelWithoutAttention,
)


class CrossLayerV2(nn.Module):
    """
    Single Cross Layer implementing equation 4 from the paper
    E_i(x_l) = x_0 ⊙ (U_l^i · g(C_l^i · g(V_l^iT x_l)) + b_l)
    """

    def __init__(
        self,
        input_dim: int,
        low_rank: Optional[int] = None,
        num_experts: int = 1,
        expert_dropout: float = 0.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.low_rank = low_rank
        self.num_experts = num_experts
        self.expert_activation = nn.ReLU()

        if low_rank is None:
            self.W = nn.Parameter(torch.empty(num_experts, input_dim, input_dim))
        else:
            self.U = nn.Parameter(torch.empty(num_experts, input_dim, low_rank))
            self.C = nn.Parameter(torch.empty(num_experts, low_rank, low_rank))
            self.V = nn.Parameter(torch.empty(num_experts, low_rank, input_dim))

        self.bias = nn.Parameter(torch.zeros(input_dim))

        # Expert gate if using multiple experts
        if num_experts > 1:
            self.expert_gate = nn.Linear(input_dim, num_experts)

        self.dropout = nn.Dropout(expert_dropout)
        self.reset_parameters()

    def reset_parameters(self):
        if self.low_rank is None:
            nn.init.xavier_uniform_(self.W)
        else:
            nn.init.xavier_uniform_(self.U)
            nn.init.xavier_uniform_(self.C)
            nn.init.xavier_uniform_(self.V)

    def get_expert_output(self, x: torch.Tensor) -> torch.Tensor:
        if self.low_rank is None:
            expert_output = torch.einsum("bi,eij,bj->bei", x, self.W, x)
        else:
            v_x = torch.einsum("eij,bj->bei", self.V, x)
            v_x = self.expert_activation(v_x)

            c_x = torch.einsum("eij,bej->bei", self.C, v_x)
            c_x = self.expert_activation(c_x)

            expert_output = torch.einsum("eij,bej->bei", self.U, c_x)

        return self.dropout(expert_output)

    def forward(self, x0: torch.Tensor, xi: torch.Tensor) -> torch.Tensor:
        expert_output = self.get_expert_output(xi)

        if self.num_experts > 1:
            gate_values = torch.softmax(self.expert_gate(xi), dim=1)
            gate_values = gate_values.unsqueeze(2)
            combined_output = torch.sum(expert_output * gate_values, dim=1)
        else:
            combined_output = expert_output.squeeze(1)

        return x0 * combined_output + self.bias + xi


class CrossNetworkV2(nn.Module):
    """
    Cross Network made up of multiple cross layers
    """

    def __init__(
        self,
        input_dim: int,
        num_layers: int,
        low_rank: Optional[int] = None,
        num_experts: int = 1,
        expert_dropout: float = 0.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_layers = num_layers

        self.cross_layers = nn.ModuleList(
            [
                CrossLayerV2(
                    input_dim=input_dim,
                    low_rank=low_rank,
                    num_experts=num_experts,
                    expert_dropout=expert_dropout,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = x
        xi = x

        for layer in self.cross_layers:
            xi = layer(x0, xi)

        return xi


class DeepCrossNetworkV2(BaseTabularModelWithoutAttention):
    def __init__(
        self,
        column_idx: Dict[str, int],
        *,
        num_cross_layers: int = 3,
        num_experts: int = 4,
        low_rank: Optional[int] = None,
        expert_dropout: float = 0.0,
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
        super(DeepCrossNetworkV2, self).__init__(
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
        self.num_experts = num_experts
        self.low_rank = low_rank
        self.expert_dropout = expert_dropout
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

        self.cross_network = CrossNetworkV2(
            input_dim=embeddings_output_dim,
            num_layers=num_cross_layers,
            low_rank=low_rank,
            num_experts=num_experts,
            expert_dropout=expert_dropout,
        )

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
