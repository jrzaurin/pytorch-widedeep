from typing import Any, Dict, List, Tuple, Optional

import torch
from torch import Tensor, nn

from pytorch_widedeep.models.tabular.mlp._layers import MLP
from pytorch_widedeep.models.tabular._base_tabular_model import (
    BaseTabularModelWithAttention,
)


class DeepFieldAwareFactorizationMachine(BaseTabularModelWithAttention):
    def __init__(
        self,
        *,
        column_idx: Dict[str, int],
        cat_embed_input: List[Tuple[str, int]],
        num_factors: int,
        reduce_sum: bool = True,
        cat_embed_dropout: Optional[float] = None,
        use_cat_bias: Optional[bool] = None,
        cat_embed_activation: Optional[str] = None,
        mlp_hidden_dims: Optional[List[int]] = None,
        mlp_activation: Optional[str] = None,
        mlp_dropout: Optional[float] = None,
        mlp_batchnorm: Optional[bool] = None,
        mlp_batchnorm_last: Optional[bool] = None,
        mlp_linear_first: Optional[bool] = None,
    ):
        super(DeepFieldAwareFactorizationMachine, self).__init__(
            column_idx=column_idx,
            input_dim=num_factors,
            cat_embed_input=cat_embed_input,
            cat_embed_dropout=cat_embed_dropout,
            use_cat_bias=use_cat_bias,
            cat_embed_activation=cat_embed_activation,
            shared_embed=False,
            add_shared_embed=None,
            frac_shared_embed=None,
            continuous_cols=None,
            cont_norm_layer=None,
            embed_continuous=None,
            embed_continuous_method=None,
            cont_embed_dropout=None,
            cont_embed_activation=None,
            quantization_setup=None,
            n_frequencies=None,
            sigma=None,
            share_last_layer=None,
            full_embed_dropout=None,
        )

        self.reduce_sum = reduce_sum

        self.mlp_hidden_dims = mlp_hidden_dims
        self.mlp_activation = mlp_activation
        self.mlp_dropout = mlp_dropout
        self.mlp_batchnorm = mlp_batchnorm
        self.mlp_batchnorm_last = mlp_batchnorm_last
        self.mlp_linear_first = mlp_linear_first

        self.n_features = len(self.column_idx)
        self.n_tokens = sum([ei[1] for ei in cat_embed_input])

        self.encoders = nn.ModuleList(
            [
                BaseTabularModelWithAttention(**config)
                for config in self._get_encoder_configs()
            ]
        )

        if self.mlp_hidden_dims is not None:
            d_hidden = [
                self.n_features * (self.n_features - 1) // 2 * num_factors
            ] + self.mlp_hidden_dims
            self.mlp = MLP(
                d_hidden=d_hidden,
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

        interactions_l: List[Tensor] = []
        for i in range(len(self.column_idx)):
            for j in range(i + 1, len(self.column_idx)):
                # the syntax [i] and [j] is to keep the shape of the tensors
                # as they are sliced within '_get_embeddings'. This will
                # return a tensor of shape (b, 1, embed_dim). Then it has to
                # be squeezed to (b, embed_dim)  before multiplied
                embed_i = self.encoders[i]._get_embeddings(X[:, [i]]).squeeze(1)
                embed_j = self.encoders[j]._get_embeddings(X[:, [j]]).squeeze(1)
                interactions_l.append(embed_i * embed_j)

        interactions = torch.cat(interactions_l, dim=1)

        if self.mlp is not None:
            interactions = interactions.view(X.size(0), -1)
            deep_out = self.mlp(interactions)
        else:
            deep_out = interactions

        if self.reduce_sum:
            deep_out = deep_out.sum(dim=1, keepdim=True)

        return deep_out

    def _get_encoder_configs(self) -> List[Dict[str, Any]]:
        config: List[Dict[str, Any]] = []
        for col, _ in self.column_idx.items():
            cat_embed_input = [(col, self.n_tokens)]
            _config = {
                "column_idx": {col: 0},
                "input_dim": self.input_dim,
                "cat_embed_input": cat_embed_input,
                "cat_embed_dropout": self.cat_embed_dropout,
                "use_cat_bias": self.use_cat_bias,
                "cat_embed_activation": self.cat_embed_activation,
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

            config.append(_config)

        return config

    @property
    def output_dim(self) -> int:
        if self.reduce_sum:
            return 1
        elif self.mlp_hidden_dims is not None:
            return self.mlp_hidden_dims[-1]
        else:
            return self.n_features * (self.n_features - 1) // 2 * self.input_dim


if __name__ == "__main__":
    import re

    import pandas as pd

    from pytorch_widedeep.models import Wide, WideDeep
    from pytorch_widedeep.datasets import load_movielens100k
    from pytorch_widedeep.preprocessing import TabPreprocessor, WidePreprocessor

    data, users, items = load_movielens100k(as_frame=True)

    list_of_genres = [
        "unknown",
        "Action",
        "Adventure",
        "Animation",
        "Children's",
        "Comedy",
        "Crime",
        "Documentary",
        "Drama",
        "Fantasy",
        "Film-Noir",
        "Horror",
        "Musical",
        "Mystery",
        "Romance",
        "Sci-Fi",
        "Thriller",
        "War",
        "Western",
    ]

    assert (
        isinstance(items, pd.DataFrame)
        and isinstance(data, pd.DataFrame)
        and isinstance(users, pd.DataFrame)
    )
    items["genre_list"] = items[list_of_genres].apply(
        lambda x: [genre for genre in list_of_genres if x[genre] == 1], axis=1
    )

    # for each element in genre_list, all to lower case, remove non-alphanumeric
    # characters, sort and join with an underscore
    def clean_genre_list(genre_list):
        return "_".join(
            sorted([re.sub(r"[^a-z0-9]", "", genre.lower()) for genre in genre_list])
        )

    items["genre_list"] = items["genre_list"].apply(clean_genre_list)

    df = pd.merge(data, users[["user_id", "age", "gender", "occupation"]], on="user_id")
    df = pd.merge(df, items[["movie_id", "genre_list"]], on="movie_id")

    df["rating"] = df["rating"].apply(lambda x: 1 if x >= 4 else 0)

    df_sample = df.sample(1000)

    tab_preprocessor = TabPreprocessor(
        cat_embed_cols=[
            "user_id",
            "movie_id",
            "age",
            "gender",
            "occupation",
            "genre_list",
        ],
        for_transformer=True,
    )

    X_tab = tab_preprocessor.fit_transform(df_sample)
    X_wide = WidePreprocessor(
        wide_cols=[
            "user_id",
            "movie_id",
            "age",
            "gender",
            "occupation",
            "genre_list",
        ]
    ).fit_transform(df_sample)

    ffm_sum = DeepFieldAwareFactorizationMachine(
        column_idx=tab_preprocessor.column_idx,
        cat_embed_input=tab_preprocessor.cat_embed_input,
        num_factors=8,
        reduce_sum=True,
        mlp_hidden_dims=[32, 16],
    )

    ffm_deep = DeepFieldAwareFactorizationMachine(
        column_idx=tab_preprocessor.column_idx,
        cat_embed_input=tab_preprocessor.cat_embed_input,
        num_factors=8,
        reduce_sum=False,
        mlp_hidden_dims=[32, 16],
    )

    X_tab_tensor = torch.tensor(X_tab)
    X_wide_tensor = torch.tensor(X_wide)

    wide = Wide(input_dim=X_tab.max())

    ffm_sum_model = WideDeep(wide=wide, deeptabular=ffm_sum)
    ffm_deep_model = WideDeep(wide=wide, deeptabular=ffm_deep)

    X_inp = {"wide": X_wide_tensor, "deeptabular": X_tab_tensor}

    res_model_sum = ffm_sum_model(X_inp)
    res_model_deep = ffm_deep_model(X_inp)
