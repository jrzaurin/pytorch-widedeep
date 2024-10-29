from typing import Dict, List, Tuple, Union, Literal, Optional

import torch
from torch import nn

from pytorch_widedeep.models.tabular.mlp._layers import MLP
from pytorch_widedeep.models.tabular._base_tabular_model import (
    BaseTabularModelWithAttention,
)


class AutoIntPlus(BaseTabularModelWithAttention):
    def __init__(
        self,
        column_idx: Dict[str, int],
        input_dim: int,
        *,
        num_heads: int = 4,
        num_layers: int = 2,
        reduction: Literal["mean", "cat"] = "mean",
        structure: Literal["stacked", "parallel"] = "parallel",
        gated: bool = True,
        cat_embed_input: Optional[List[Tuple[str, int]]],
        cat_embed_dropout: Optional[float],
        use_cat_bias: Optional[bool],
        cat_embed_activation: Optional[str],
        shared_embed: Optional[bool],
        add_shared_embed: Optional[bool],
        frac_shared_embed: Optional[float],
        continuous_cols: Optional[List[str]],
        cont_norm_layer: Optional[Literal["batchnorm", "layernorm"]],
        embed_continuous_method: Optional[Literal["standard", "piecewise", "periodic"]],
        cont_embed_dropout: Optional[float],
        cont_embed_activation: Optional[str],
        quantization_setup: Optional[Dict[str, List[float]]],
        n_frequencies: Optional[int],
        sigma: Optional[float],
        share_last_layer: Optional[bool],
        full_embed_dropout: Optional[bool],
        mlp_hidden_dims: List[int] = [200, 100],
        mlp_activation: str = "relu",
        mlp_dropout: Union[float, List[float]] = 0.1,
        mlp_batchnorm: bool = False,
        mlp_batchnorm_last: bool = False,
        mlp_linear_first: bool = True,
    ):
        super(AutoIntPlus, self).__init__(
            column_idx=column_idx,
            input_dim=input_dim,
            cat_embed_input=cat_embed_input,
            cat_embed_dropout=cat_embed_dropout,
            use_cat_bias=use_cat_bias,
            cat_embed_activation=cat_embed_activation,
            shared_embed=shared_embed,
            add_shared_embed=add_shared_embed,
            frac_shared_embed=frac_shared_embed,
            continuous_cols=continuous_cols,
            cont_norm_layer=cont_norm_layer,
            embed_continuous_method=embed_continuous_method,
            cont_embed_dropout=cont_embed_dropout,
            cont_embed_activation=cont_embed_activation,
            quantization_setup=quantization_setup,
            n_frequencies=n_frequencies,
            sigma=sigma,
            share_last_layer=share_last_layer,
            full_embed_dropout=full_embed_dropout,
        )

        self.input_dim = input_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.reduction = reduction
        self.structure = structure
        self.gated = gated

        self.attention_layers = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    embed_dim=input_dim,
                    num_heads=num_heads,
                    batch_first=True,
                )
                for _ in range(num_layers)
            ]
        )

        if self.structure == "parallel":
            deep_network_inp_dim = input_dim * len(column_idx)
            if self.gated:
                self.gate = nn.Linear(input_dim + mlp_hidden_dims[-1], input_dim)
        else:  # structure == "stacked"
            if self.reduction == "mean":
                deep_network_inp_dim = input_dim
            else:
                deep_network_inp_dim = input_dim * len(column_idx)

        mlp_hidden_dims = [deep_network_inp_dim] + mlp_hidden_dims
        self.deep_network = MLP(
            mlp_hidden_dims,
            mlp_activation,
            mlp_dropout,
            mlp_batchnorm,
            mlp_batchnorm_last,
            mlp_linear_first,
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        x = self._get_embeddings(X)
        attn_output = self._apply_attention_layers(x)
        reduced_attn_output = self._reduce_attention_output(attn_output)
        if self.structure == "parallel":
            return self._parallel_output(reduced_attn_output, x)
        else:  # structure == "stacked"
            return self.deep_network(reduced_attn_output)

    def _apply_attention_layers(self, x: torch.Tensor) -> torch.Tensor:
        attn_output = x
        for layer in self.attention_layers:
            layer_out, _ = layer(attn_output, attn_output, attn_output)
            attn_output = layer_out + attn_output
        return attn_output

    def _reduce_attention_output(self, attn_output: torch.Tensor) -> torch.Tensor:
        return (
            attn_output.mean(dim=1)
            if self.reduction == "mean"
            else attn_output.flatten(1)
        )

    def _parallel_output(
        self, attn_output: torch.Tensor, x: torch.Tensor
    ) -> torch.Tensor:
        deep_output = self.deep_network(x)

        if not self.gated:
            return torch.cat([attn_output, deep_output], dim=1)

        combined_features = torch.cat([attn_output, deep_output], dim=-1)
        gate_weights = torch.sigmoid(self.gate(combined_features))
        return gate_weights * attn_output + (1 - gate_weights) * deep_output

    @property
    def output_dim(self) -> int:
        if self.structure == "parallel":
            if self.gated:
                return self.deep_network.output_dim
            else:
                return self.deep_network.output_dim + self.input_dim * len(
                    self.column_idx
                )
        else:
            return self.deep_network.output_dim
