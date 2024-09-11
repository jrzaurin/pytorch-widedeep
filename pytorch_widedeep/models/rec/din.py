from typing import Dict, List, Tuple, Literal, Optional

import torch

from pytorch_widedeep.models.rec._layers import ActivationUnit
from pytorch_widedeep.models.tabular.mlp._layers import MLP
from pytorch_widedeep.models.tabular._base_tabular_model import (
    BaseTabularModelWithAttention,
)


class DeepInterestNetwork(BaseTabularModelWithAttention):
    def __init__(
        self,
        column_idx: Dict[str, int],
        input_dim: int,
        item_col: str,
        user_behavior_cols: List[str],
        activation: Literal["prelu", "dice"],
        padding_idx: int,
        cat_embed_input: List[Tuple[str, int]],
        *,
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
        super(DeepInterestNetwork, self).__init__(
            column_idx=column_idx,
            input_dim=input_dim,
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

        self.item_col = item_col
        self.user_behavior_cols = user_behavior_cols
        self.activation = activation
        self.padding_idx = padding_idx

        # Assuming that the user behavior columns are lists of items the user
        # interacted with, they should not contribute to the total number of
        # unique feature values (tokens if you want)
        for col in user_behavior_cols:
            assert (col, 0) in cat_embed_input

        self.mlp_hidden_dims = mlp_hidden_dims
        self.mlp_activation = mlp_activation
        self.mlp_dropout = mlp_dropout
        self.mlp_batchnorm = mlp_batchnorm
        self.mlp_batchnorm_last = mlp_batchnorm_last
        self.mlp_linear_first = mlp_linear_first

        self.item_col_idx = self.column_idx[item_col]
        self.user_behavior_col_idx = [
            self.column_idx[col] for col in user_behavior_cols
        ]
        self.other_cols_idx = [
            self.column_idx[col]
            for col in self.columns
            if col not in user_behavior_cols + [item_col]
        ]

        self.activation = activation

        self.attention = ActivationUnit(input_dim, activation)

        if self.mlp_hidden_dims is not None:
            mlp_input_dim = input_dim * 3 if self.other_cols_idx else input_dim * 2
            self.mlp = MLP(
                d_hidden=[mlp_input_dim] + self.mlp_hidden_dims,
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

    def forward(self, X: torch.Tensor) -> torch.Tensor:

        X_emb = self._get_embeddings(X)

        item_embed = X_emb[:, self.item_col_idx, :]
        user_behavior_embed = X_emb[:, self.user_behavior_col_idx, :]
        other_features_embed = X_emb[:, self.other_cols_idx, :]

        mask = (
            (X[:, self.user_behavior_col_idx, :] != self.padding_idx)
            .float()
            .to(X.device)
        )
        attention_scores = self.attention(item_embed, user_behavior_embed)
        attention_scores = attention_scores * mask

        user_interest = torch.sum(
            attention_scores.unsqueeze(-1) * user_behavior_embed, dim=1
        )

        concat_feature = torch.cat(
            [item_embed, user_interest, other_features_embed], dim=1
        )

        if self.mlp is not None:
            concat_feature = self.mlp(concat_feature)

        return concat_feature
