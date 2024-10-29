from typing import Dict, List, Tuple, Literal, Optional

import torch
from torch import nn

from pytorch_widedeep.models.tabular._base_tabular_model import (
    BaseTabularModelWithAttention,
)


class AutoInt(BaseTabularModelWithAttention):
    def __init__(
        self,
        column_idx: Dict[str, int],
        input_dim: int,
        *,
        num_heads: int = 4,
        num_layers: int = 2,
        reduction: Literal["mean", "cat"] = "mean",
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
    ):
        super(AutoInt, self).__init__(
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

        self.num_heads = num_heads
        self.num_layers = num_layers
        self.reduction = reduction

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

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        x = self._get_embeddings(X)

        for layer in self.attention_layers:
            attn_output, _ = layer(x, x, x)
            x = attn_output + x

        if self.reduction == "mean":
            out = x.mean(dim=1)
        else:
            out = x.view(x.size(0), -1)

        return out

    @property
    def output_dim(self) -> int:
        if self.reduction == "mean":
            return self.input_dim
        else:
            return self.input_dim * len(self.column_idx)
