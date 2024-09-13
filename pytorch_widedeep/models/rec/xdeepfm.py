from typing import Dict, List, Tuple, Optional

import torch
from torch import Tensor

from pytorch_widedeep.models.rec._layers import CompressedInteractionNetwork
from pytorch_widedeep.models.tabular.mlp._layers import MLP
from pytorch_widedeep.models.tabular._base_tabular_model import (
    BaseTabularModelWithAttention,
)


class xDeepFM(BaseTabularModelWithAttention):
    def __init__(
        self,
        *,
        column_idx: Dict[str, int],
        cat_embed_input: List[Tuple[str, int]],
        input_dim: int,
        cin_layer_dims: List[int],
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
        super(xDeepFM, self).__init__(
            column_idx=column_idx,
            input_dim=input_dim,
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

        self.mlp_hidden_dims = mlp_hidden_dims
        self.mlp_activation = mlp_activation
        self.mlp_dropout = mlp_dropout
        self.mlp_batchnorm = mlp_batchnorm
        self.mlp_batchnorm_last = mlp_batchnorm_last
        self.mlp_linear_first = mlp_linear_first

        self.reduce_sum = reduce_sum
        self.cin_layer_dims = cin_layer_dims

        self.n_features = len(self.column_idx)

        self.cin = CompressedInteractionNetwork(
            num_cols=self.n_features, cin_layer_dims=self.cin_layer_dims
        )

        if self.mlp_hidden_dims is not None:
            if (
                self.mlp_hidden_dims[-1] != sum(self.cin_layer_dims)
                and not self.reduce_sum
            ):
                d_hidden = (
                    [sum(self.cin_layer_dims)]
                    + self.mlp_hidden_dims
                    + [sum(self.cin_layer_dims)]
                )
            else:
                d_hidden = [sum(self.cin_layer_dims)] + self.mlp_hidden_dims

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

        embeddings = self._get_embeddings(X)
        cin_out = self.cin(embeddings)

        if self.mlp is None:
            if self.reduce_sum:
                return cin_out.sum(dim=1, keepdim=True)
            return cin_out

        mlp_out = self.mlp(cin_out)

        if self.reduce_sum:
            cin_out = cin_out.sum(dim=1, keepdim=True)
            mlp_out = mlp_out.sum(dim=1, keepdim=True)

        return mlp_out + cin_out

    @property
    def output_dim(self):
        if self.reduce_sum:
            return 1
        else:
            return sum(self.cin_layer_dims)


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

    xdfm_sum = xDeepFM(
        column_idx=tab_preprocessor.column_idx,
        cat_embed_input=tab_preprocessor.cat_embed_input,
        input_dim=8,
        cin_layer_dims=[16, 8],
        reduce_sum=True,
        mlp_hidden_dims=[32, 16],
    )

    xdfm_deep = xDeepFM(
        column_idx=tab_preprocessor.column_idx,
        cat_embed_input=tab_preprocessor.cat_embed_input,
        input_dim=8,
        cin_layer_dims=[16, 8],
        reduce_sum=False,
        mlp_hidden_dims=[32, 16],
    )

    X_tab_tensor = torch.tensor(X_tab)
    X_wide_tensor = torch.tensor(X_wide)

    wide = Wide(input_dim=X_tab.max())

    xdfm_deep_model = WideDeep(wide=wide, deeptabular=xdfm_sum)
    xdfm_deep_model = WideDeep(wide=wide, deeptabular=xdfm_deep)

    X_inp = {"wide": X_wide_tensor, "deeptabular": X_tab_tensor}

    res_model_sum = xdfm_deep_model(X_inp)
    res_model_deep = xdfm_deep_model(X_inp)
