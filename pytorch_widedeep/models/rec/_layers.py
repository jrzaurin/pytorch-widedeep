from typing import Dict, List, Tuple, Literal, Optional

from torch import Tensor

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
