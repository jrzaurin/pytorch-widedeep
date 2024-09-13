from typing import Any, Dict, List, Tuple, Literal, Optional

import torch
from torch import Tensor, nn

from pytorch_widedeep.models.rec._layers import ActivationUnit
from pytorch_widedeep.models.tabular.mlp._layers import MLP
from pytorch_widedeep.models._base_wd_model_component import BaseWDModelComponent
from pytorch_widedeep.models.tabular._base_tabular_model import (
    BaseTabularModelWithAttention,
    BaseTabularModelWithoutAttention,
)


class DeepInterestNetwork(BaseWDModelComponent):
    def __init__(
        self,
        *,
        column_idx: Dict[str, int],
        target_item_col: str,
        user_behavior_confiq: Tuple[List[str], int, int],
        rating_seq_config: Tuple[List[str], int, int],
        other_seq_cols_confiq: List[Tuple[List[str], int, int]],
        other_cols_config: List[Tuple[str, int, int]],
        activation: Literal["prelu", "dice"] = "prelu",
        cat_embed_dropout: Optional[float] = None,
        use_cat_bias: Optional[bool] = None,
        cat_embed_activation: Optional[str] = None,
        continuous_cols: Optional[List[str]] = None,
        cont_norm_layer: Optional[Literal["batchnorm", "layernorm"]] = None,
        embed_continuous: Optional[bool] = None,
        embed_continuous_method: Optional[
            Literal["piecewise", "periodic"]
        ] = "piecewise",
        cont_embed_dim: Optional[int] = None,
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
        super(DeepInterestNetwork, self).__init__()

        self.column_idx = {
            k: v for k, v in sorted(column_idx.items(), key=lambda x: x[1])
        }

        self.cat_embed_dropout = cat_embed_dropout
        self.use_cat_bias = use_cat_bias
        self.cat_embed_activation = cat_embed_activation
        self.continuous_cols = continuous_cols
        self.cont_norm_layer = cont_norm_layer
        self.embed_continuous = embed_continuous
        self.embed_continuous_method = embed_continuous_method
        self.cont_embed_dim = cont_embed_dim
        self.cont_embed_dropout = cont_embed_dropout
        self.cont_embed_activation = cont_embed_activation
        self.quantization_setup = quantization_setup
        self.n_frequencies = n_frequencies
        self.sigma = sigma
        self.share_last_layer = share_last_layer
        self.full_embed_dropout = full_embed_dropout

        self.activation = activation

        self.mlp_hidden_dims = mlp_hidden_dims
        self.mlp_activation = mlp_activation
        self.mlp_dropout = mlp_dropout
        self.mlp_batchnorm = mlp_batchnorm
        self.mlp_batchnorm_last = mlp_batchnorm_last
        self.mlp_linear_first = mlp_linear_first

        self.target_item_col = target_item_col
        self.target_item_idx = self.column_idx[target_item_col]

        self.user_behavior_indexes = [
            self.column_idx[col] for col in user_behavior_confiq[0]
        ]
        self.user_behavior_embed = BaseTabularModelWithAttention(
            **self._get_seq_cols_embed_confiq(user_behavior_confiq)
        )

        self.rating_seq_indexes = [self.column_idx[col] for col in rating_seq_config[0]]
        self.rating_embed = BaseTabularModelWithAttention(
            **self._get_seq_cols_embed_confiq(rating_seq_config)
        )

        self.other_seq_indexes: Dict[str, List[int]] = {}
        for i, el in enumerate(other_seq_cols_confiq):
            key = f"seq_{i}"
            idxs = [self.column_idx[col] for col in el[0]]
            self.other_seq_indexes[key] = idxs
        other_seq_cols_config = {
            f"seq_{i}": self._get_seq_cols_embed_confiq(el)
            for i, el in enumerate(other_seq_cols_confiq)
        }
        self.other_seq_cols_embed = nn.ModuleDict(
            {
                key: BaseTabularModelWithAttention(**config)
                for key, config in other_seq_cols_config.items()
            }
        )

        self.other_cols_idx = [
            self.column_idx[col] for col in [el[0] for el in other_cols_config]
        ]
        self.other_col_embed = BaseTabularModelWithoutAttention(
            **self._get_other_cols_embed_config(other_cols_config)
        )

        self.attention = ActivationUnit(user_behavior_confiq[2], activation)

        if self.mlp_hidden_dims is not None:
            mlp_input_dim = (
                user_behavior_confiq[2] * 2  # item_embed + user_interest
                + sum([el[2] for el in other_seq_cols_confiq])  # other_seq_embed
                + sum([el[2] for el in other_cols_config])  # other_cols_embed
            )
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

    def forward(self, X: Tensor) -> Tensor:

        X_target_item = X[:, [self.target_item_idx]]

        X_user_behavior = X[:, self.user_behavior_indexes]
        # 0 is considered the padding index
        mask = (X_user_behavior != 0).float().to(X.device)

        X_rating = X[:, self.rating_seq_indexes]
        X_other_seq: Dict[str, Tensor] = {
            col: X[:, idx] for col, idx in self.other_seq_indexes.items()
        }
        X_other_cols = X[:, self.other_cols_idx]

        item_embed = self.user_behavior_embed.cat_embed.embed(X_target_item.long())
        user_behavior_embed = self.user_behavior_embed._get_embeddings(X_user_behavior)
        rating_embed = self.rating_embed._get_embeddings(X_rating)
        other_seq_embed = torch.cat(
            [
                self.other_seq_cols_embed[col]._get_embeddings(X_other_seq[col])
                for col in self.other_seq_indexes.keys()
            ]
        ).sum(1)
        other_cols_embed = self.other_col_embed._get_embeddings(X_other_cols)

        user_behavior_embed_with_rating = user_behavior_embed * rating_embed

        attention_scores = self.attention(item_embed, user_behavior_embed_with_rating)
        attention_scores = attention_scores * mask

        user_interest = (
            attention_scores.unsqueeze(-1) * user_behavior_embed_with_rating
        ).sum(1)

        deep_out = torch.cat(
            [item_embed.squeeze(1), user_interest, other_seq_embed, other_cols_embed],
            dim=1,
        )

        if self.mlp is not None:
            deep_out = self.mlp(deep_out)

        return deep_out

    @staticmethod
    def _get_seq_cols_embed_confiq(tup: Tuple[List[str], int, int]) -> Dict[str, Any]:
        # tup[0] is the list of columns
        # tup[1] is the number of unique feat value or "n_tokens"
        # tup[2] is the embedding dimension

        # Once sliced, the indexes will go from 0 to len(tup[0])
        column_idx = {col: i for i, col in enumerate(tup[0])}

        # This is a hack so that I can use any BaseTabularModelWithAttention.
        # For this model to work 'cat_embed_input' is normally a List of
        # Tuples where the first element is the column name and the second is
        # the number of unique values for that column. That second elements
        # is added internally to compute what one could call "n_tokens". Here
        # I'm passing that value as the second element of the first tuple
        # and then adding 0s for the rest of the columns
        cat_embed_input = [(tup[0][0], tup[1])] + [(col, 0) for col in tup[0][1:]]

        input_dim = tup[2]

        col_config = {
            "column_idx": column_idx,
            "input_dim": input_dim,
            "cat_embed_input": cat_embed_input,
            "cat_embed_dropout": None,
            "use_cat_bias": None,
            "cat_embed_activation": None,
            "shared_embed": None,
            "add_shared_embed": None,
            "frac_shared_embed": None,
            "continuous_cols": None,
            "cont_norm_layer": None,
            "embed_continuous": None,
            "embed_continuous_method": None,
            "cont_embed_dropout": None,
            "cont_embed_activation": None,
            "quantization_setup": None,
            "n_frequencies": None,
            "sigma": None,
            "share_last_layer": None,
            "full_embed_dropout": None,
        }

        return col_config

    def _get_other_cols_embed_config(
        self, tups: List[Tuple[str, int, int]]
    ) -> Dict[str, Any]:
        cols_config = {
            "column_idx": {col: i for i, col in enumerate([el[0] for el in tups])},
            "cat_embed_input": tups,
            "cat_embed_dropout": self.cat_embed_dropout,
            "use_cat_bias": self.use_cat_bias,
            "cat_embed_activation": self.cat_embed_activation,
            "continuous_cols": self.continuous_cols,
            "cont_norm_layer": self.cont_norm_layer,
            "embed_continuous": self.embed_continuous,
            "embed_continuous_method": self.embed_continuous_method,
            "cont_embed_dim": self.cont_embed_dim,
            "cont_embed_dropout": self.cont_embed_dropout,
            "cont_embed_activation": self.cont_embed_activation,
            "quantization_setup": self.quantization_setup,
            "n_frequencies": self.n_frequencies,
            "sigma": self.sigma,
            "share_last_layer": self.share_last_layer,
            "full_embed_dropout": self.full_embed_dropout,
        }

        return cols_config
