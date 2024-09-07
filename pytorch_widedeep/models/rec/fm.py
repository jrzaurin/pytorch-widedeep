from typing import Any, Dict, List, Tuple, Literal, Optional

import torch
from torch import Tensor, nn

from pytorch_widedeep.models._base_wd_model_component import BaseWDModelComponent
from pytorch_widedeep.models.tabular._base_tabular_model import (
    BaseTabularModelWithAttention as FeatureEncodeer,
)


def factorization_machine(input: Tensor, weights: Optional[Tensor] = None) -> Tensor:
    if weights is None:
        w = torch.ones(input.size()).to(input.device)
    else:
        w = weights

    square_of_sum = (input * w).sum(dim=1) ** 2.0
    sum_of_square = (input**2.0 * w**2.0).sum(dim=1)

    return (0.5 * (square_of_sum - sum_of_square)).sum(1, keepdim=True)


class QuasiLinearEncoder(FeatureEncodeer):
    def __init__(
        self,
        column_idx: Dict[str, int],
        *,
        cat_embed_input: Optional[List[Tuple[str, int]]],
        continuous_cols: Optional[List[str]],
        cont_norm_layer: Optional[Literal["batchnorm", "layernorm"]],
        embed_continuous_method: Optional[Literal["piecewise", "periodic"]],
        quantization_setup: Optional[Dict[str, List[float]]],
        n_frequencies: Optional[int],
        sigma: Optional[float],
        share_last_layer: Optional[bool],
    ):
        super(QuasiLinearEncoder, self).__init__(
            column_idx=column_idx,
            input_dim=1,
            cat_embed_input=cat_embed_input,
            cat_embed_dropout=None,
            use_cat_bias=False,
            cat_embed_activation=None,
            shared_embed=False,
            add_shared_embed=None,
            frac_shared_embed=None,
            continuous_cols=continuous_cols,
            cont_norm_layer=cont_norm_layer,
            embed_continuous=True,
            embed_continuous_method=embed_continuous_method,
            cont_embed_dropout=None,
            cont_embed_activation=None,
            quantization_setup=quantization_setup,
            n_frequencies=n_frequencies,
            sigma=sigma,
            share_last_layer=share_last_layer,
            full_embed_dropout=None,
        )

        self.bias = torch.nn.Parameter(torch.zeros(1))

    def forward(self, X: Tensor) -> Tensor:
        embed = self._get_embeddings(X)
        return embed.sum(dim=1) + self.bias

    @property
    def output_dim(self) -> int:
        return 1


class FactorizationMachine(FeatureEncodeer):
    def __init__(
        self,
        column_idx: Dict[str, int],
        num_factors: int,
        *,
        col_weights: Optional[Dict[str, float]] = None,
        cat_embed_input: Optional[List[Tuple[str, int]]] = None,
        cat_embed_dropout: Optional[float] = None,
        use_cat_bias: Optional[bool] = None,
        cat_embed_activation: Optional[str] = None,
        continuous_cols: Optional[List[str]] = None,
        cont_norm_layer: Optional[Literal["batchnorm", "layernorm"]] = None,
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
    ):
        super(FactorizationMachine, self).__init__(
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
            embed_continuous=True,
            embed_continuous_method=embed_continuous_method,
            cont_embed_dropout=cont_embed_dropout,
            cont_embed_activation=cont_embed_activation,
            quantization_setup=quantization_setup,
            n_frequencies=n_frequencies,
            sigma=sigma,
            share_last_layer=share_last_layer,
            full_embed_dropout=full_embed_dropout,
        )

        self.col_weights = col_weights

        self.linear = QuasiLinearEncoder(
            column_idx=column_idx,
            cat_embed_input=cat_embed_input,
            continuous_cols=continuous_cols,
            cont_norm_layer=cont_norm_layer,
            embed_continuous_method=embed_continuous_method,
            quantization_setup=quantization_setup,
            n_frequencies=n_frequencies,
            sigma=sigma,
            share_last_layer=share_last_layer,
        )

    def forward(self, X: Tensor) -> Tensor:

        # TO DO: Adjust X to be used as a linear component (all embeddings from 1 to max)
        linear_output = self.linear(X)

        embed = self._get_embeddings(X)
        if self.col_weights is not None:
            col_weights = torch.tensor(
                [self.col_weights[col] for col in self.column_idx.keys()]
            ).to(embed.device)
        else:
            col_weights = None

        factorization_output = factorization_machine(embed, col_weights)

        return linear_output + factorization_output

    @property
    def output_dim(self) -> int:
        return 1


class FieldAwareFactorizationMachine(BaseWDModelComponent):
    def __init__(
        self,
        column_idx: Dict[str, int],
        num_factors: int,
        *,
        cat_embed_input: Optional[List[Tuple[str, int]]] = None,
        cat_embed_dropout: Optional[float] = None,
        use_cat_bias: Optional[bool] = None,
        cat_embed_activation: Optional[str] = None,
        continuous_cols: Optional[List[str]] = None,
        cont_norm_layer: Optional[Literal["batchnorm", "layernorm"]] = None,
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
    ):
        super(FieldAwareFactorizationMachine, self).__init__()

        self.column_idx = column_idx
        self.num_factors = num_factors
        self.cat_embed_input = cat_embed_input
        self.cat_embed_dropout = cat_embed_dropout
        self.use_cat_bias = use_cat_bias
        self.cat_embed_activation = cat_embed_activation
        self.continuous_cols = continuous_cols
        self.cont_norm_layer = cont_norm_layer
        self.embed_continuous_method = embed_continuous_method
        self.cont_embed_dropout = cont_embed_dropout
        self.cont_embed_activation = cont_embed_activation
        self.quantization_setup = quantization_setup
        self.n_frequencies = n_frequencies
        self.sigma = sigma
        self.share_last_layer = share_last_layer
        self.full_embed_dropout = full_embed_dropout

        self.linear = QuasiLinearEncoder(
            column_idx=column_idx,
            cat_embed_input=cat_embed_input,
            continuous_cols=continuous_cols,
            cont_norm_layer=cont_norm_layer,
            embed_continuous_method=embed_continuous_method,
            quantization_setup=quantization_setup,
            n_frequencies=n_frequencies,
            sigma=sigma,
            share_last_layer=share_last_layer,
        )

        self.feature_encoders = nn.ModuleDict(
            {
                col: FeatureEncodeer(**config)
                for col, config in self._encoders_configs().items()
            }
        )

    def forward(self, X: Tensor) -> Tensor:

        # TO DO: Adjust X to be used as a linear component (all embeddings from 1 to max)
        linear_output = self.linear(X)

        embeds = []
        for col, idx in self.column_idx.items():
            embeds.append(self.feature_encoders[col]._get_embeddings(X[:, [idx]]))

        # embeds = [
        #     self.feature_encoders[col]._get_embeddings(X[:, [idx]])
        #     for col, idx in self.column_idx.items()
        # ]

        interactions_l: List[Tensor] = []
        for i in range(len(self.column_idx)):
            for j in range(i + 1, len(self.column_idx)):
                interactions_l.append((embeds[j][:, i] * embeds[i][:, j]))

        interactions = torch.stack(interactions_l).sum(dim=1).sum(dim=1, keepdim=True)

        return linear_output + interactions

    def _encoders_configs(self) -> Dict[str, Any]:
        sorted_column_idx = dict(sorted(self.column_idx.items(), key=lambda x: x[1]))

        if self.cat_embed_input is not None:
            cat_cols: Optional[List[str]] = [el[0] for el in self.cat_embed_input]
        else:
            cat_cols = None

        encoder_configs: Dict[str, Any] = {}
        for col, _ in sorted_column_idx.items():
            if cat_cols is None or (cat_cols is not None and col not in cat_cols):
                # col is continuous
                encoder_configs[col] = {
                    "column_idx": {col: 0},
                    "input_dim": self.num_factors,
                    "cat_embed_input": None,
                    "cat_embed_dropout": None,
                    "use_cat_bias": False,
                    "cat_embed_activation": None,
                    "shared_embed": False,
                    "add_shared_embed": None,
                    "frac_shared_embed": None,
                    "continuous_cols": [col],
                    "cont_norm_layer": self.cont_norm_layer,
                    "embed_continuous": True,
                    "embed_continuous_method": self.embed_continuous_method,
                    "cont_embed_dropout": self.cont_embed_dropout,
                    "cont_embed_activation": self.cont_embed_activation,
                    "quantization_setup": self.quantization_setup,
                    "n_frequencies": self.n_frequencies,
                    "sigma": self.sigma,
                    "share_last_layer": self.share_last_layer,
                    "full_embed_dropout": self.full_embed_dropout,
                }
            else:
                # col is categorical
                encoder_configs[col] = {
                    "column_idx": {col: 0},
                    "input_dim": self.num_factors,
                    "cat_embed_input": [self.cat_embed_input[cat_cols.index(col)]],
                    "cat_embed_dropout": self.cat_embed_dropout,
                    "use_cat_bias": self.use_cat_bias,
                    "cat_embed_activation": self.cat_embed_activation,
                    "shared_embed": False,
                    "add_shared_embed": None,
                    "frac_shared_embed": None,
                    "continuous_cols": None,
                    "cont_norm_layer": None,
                    "embed_continuous": False,
                    "embed_continuous_method": None,
                    "cont_embed_dropout": None,
                    "cont_embed_activation": None,
                    "quantization_setup": None,
                    "n_frequencies": None,
                    "sigma": None,
                    "share_last_layer": None,
                    "full_embed_dropout": None,
                }

        return encoder_configs

    @property
    def output_dim(self) -> int:
        return 1
