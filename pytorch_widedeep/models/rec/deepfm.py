from typing import Dict, List, Tuple, Literal, Optional

from torch import Tensor

from pytorch_widedeep.models.tabular.mlp._layers import MLP
from pytorch_widedeep.models.tabular._base_tabular_model import (
    BaseTabularModelWithAttention,
)


def factorization_machine(input: Tensor, reduce_sum: bool = True) -> Tensor:
    square_of_sum = (input).sum(dim=1) ** 2.0
    sum_of_square = (input**2.0).sum(dim=1)
    if reduce_sum:
        return 0.5 * (square_of_sum - sum_of_square).sum(1, keepdim=True)
    else:
        return 0.5 * (square_of_sum - sum_of_square)


class DeepFactorizationMachine(BaseTabularModelWithAttention):
    def __init__(
        self,
        *,
        column_idx: Dict[str, int],
        num_factors: int,
        reduce_sum: bool = True,
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

        self.reduce_sum = reduce_sum

        self.mlp_hidden_dims = mlp_hidden_dims
        self.mlp_activation = mlp_activation
        self.mlp_dropout = mlp_dropout
        self.mlp_batchnorm = mlp_batchnorm
        self.mlp_batchnorm_last = mlp_batchnorm_last
        self.mlp_linear_first = mlp_linear_first

        if self.mlp_hidden_dims is not None:

            if self.mlp_hidden_dims[-1] != self.input_dim:
                d_hidden = (
                    [self.input_dim * len(self.column_idx)]
                    + self.mlp_hidden_dims
                    + [self.input_dim]
                )
            else:
                d_hidden = [
                    self.input_dim * len(self.column_idx)
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
        embed = self._get_embeddings(X)
        fm_output = factorization_machine(embed, self.reduce_sum)

        if self.mlp is None:
            return fm_output

        mlp_input = embed.view(embed.size(0), -1)
        mlp_output = self.mlp(mlp_input)

        if self.reduce_sum:
            mlp_output = mlp_output.sum(1, keepdim=True)

        return fm_output + mlp_output

    @property
    def output_dim(self) -> int:
        if self.reduce_sum:
            return 1
        else:
            return self.input_dim


if __name__ == "__main__":
    import re

    import torch
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

    fm_sum = DeepFactorizationMachine(
        column_idx=tab_preprocessor.column_idx,
        cat_embed_input=tab_preprocessor.cat_embed_input,
        num_factors=8,
        mlp_hidden_dims=[32, 16],
    )

    fm_deep = DeepFactorizationMachine(
        column_idx=tab_preprocessor.column_idx,
        cat_embed_input=tab_preprocessor.cat_embed_input,
        reduce_sum=False,
        num_factors=8,
        mlp_hidden_dims=[32, 16],
    )

    X_tab_tensor = torch.tensor(X_tab)
    X_wide_tensor = torch.tensor(X_wide)

    wide = Wide(input_dim=X_tab.max())

    fm_sum_model = WideDeep(wide=wide, deeptabular=fm_sum)
    fm_deep_model = WideDeep(wide=wide, deeptabular=fm_deep)

    X_inp = {"wide": X_wide_tensor, "deeptabular": X_tab_tensor}

    res_model_sum = fm_sum_model(X_inp)
    res_model_deep = fm_deep_model(X_inp)
