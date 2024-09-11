from typing import Dict, List, Tuple, Literal, Optional

from torch import Tensor

from pytorch_widedeep.models.rec._layers import PseudoLinear
from pytorch_widedeep.models.tabular.mlp._layers import MLP
from pytorch_widedeep.models.tabular._base_tabular_model import (
    BaseTabularModelWithAttention,
)


def factorization_machine(input: Tensor) -> Tensor:
    square_of_sum = (input).sum(dim=1) ** 2.0
    sum_of_square = (input**2.0).sum(dim=1)
    return 0.5 * (square_of_sum - sum_of_square).sum(1, keepdim=True)


class DeepFactorizationMachine(BaseTabularModelWithAttention):
    def __init__(
        self,
        column_idx: Dict[str, int],
        num_factors: int,
        with_pseudo_linear: bool,
        *,
        cat_embed_input: Optional[List[Tuple[str, int]]] = None,
        cat_embed_dropout: Optional[float] = None,
        use_cat_bias: Optional[bool] = None,
        cat_embed_activation: Optional[str] = None,
        continuous_cols: Optional[List[str]] = None,
        cont_norm_layer: Optional[Literal["batchnorm", "layernorm"]] = None,
        embed_continuous: Optional[bool] = None,
        embed_continuous_method: Optional[
            Literal["piecewise", "periodic"]
        ] = "piecewise",
        cont_embed_dropout: Optional[float] = None,
        cont_embed_activation: Optional[str] = None,
        quantization_setup: Optional[Dict[str, List[float]]] = None,
        n_frequencies: Optional[int] = None,
        sigma: Optional[float] = None,
        share_last_layer: Optional[bool] = None,
        full_embed_dropout: Optional[bool] = None,
        mlp_hidden_dims: Optional[List[int]] = None,
        mlp_activation: Optional[str] = None,
        mlp_dropout: Optional[float] = None,
        mlp_batchnorm: Optional[bool] = None,
        mlp_batchnorm_last: Optional[bool] = None,
        mlp_linear_first: Optional[bool] = None,
    ):
        super(DeepFactorizationMachine, self).__init__(
            column_idx=column_idx,
            input_dim=num_factors,
            cat_embed_input=cat_embed_input,
            cat_embed_dropout=cat_embed_dropout,
            use_cat_bias=use_cat_bias,
            cat_embed_activation=cat_embed_activation,
            shared_embed=False,
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

        self.mlp_hidden_dims = mlp_hidden_dims
        self.mlp_activation = mlp_activation
        self.mlp_dropout = mlp_dropout
        self.mlp_batchnorm = mlp_batchnorm
        self.mlp_batchnorm_last = mlp_batchnorm_last
        self.mlp_linear_first = mlp_linear_first

        if with_pseudo_linear:
            self.linear = PseudoLinear(
                column_idx=column_idx,
                cat_embed_input=cat_embed_input,
                cat_embed_dropout=cat_embed_dropout,
                use_cat_bias=use_cat_bias,
                cat_embed_activation=cat_embed_activation,
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
        else:
            self.linear = None

        if self.mlp_hidden_dims is not None:
            self.mlp = MLP(
                d_hidden=[self.input_dim * len(self.column_idx)] + self.mlp_hidden_dims,
                activation=(
                    "relu" if self.mlp_activation is None else self.mlp_activation
                ),
                dropout=0.0 if self.mlp_dropout is None else self.mlp_dropout,
                batchnorm=False if self.mlp_batchnorm is None else self.mlp_batchnorm,
                batchnorm_last=(
                    False
                    if self.mlp_batchnorm_last is None
                    else self.mlp_batchnorm_last
                ),
                linear_first=(
                    False if self.mlp_linear_first is None else self.mlp_linear_first
                ),
            )
        else:
            self.mlp = None

    def forward(self, X: Tensor) -> Tensor:

        embed = self._get_embeddings(X)
        factorization_output = factorization_machine(embed)

        if self.mlp is not None:
            mlp_input = embed.view(embed.size(0), -1)
            mlp_output = self.mlp(mlp_input)
            deep_out = factorization_output + mlp_output
        else:
            deep_out = factorization_output

        if self.linear is not None:
            return self.linear(X) + deep_out
        else:
            return deep_out

    @property
    def output_dim(self) -> int:
        return 1
