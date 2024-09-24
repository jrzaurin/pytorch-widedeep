import re
from typing import List, Tuple

import pandas as pd

from pytorch_widedeep import Trainer
from pytorch_widedeep.models import Wide, WideDeep
from pytorch_widedeep.metrics import Accuracy
from pytorch_widedeep.datasets import load_movielens100k
from pytorch_widedeep.models.rec import (
    DeepFactorizationMachine,
    ExtremeDeepFactorizationMachine,
    DeepFieldAwareFactorizationMachine,
)
from pytorch_widedeep.preprocessing import TabPreprocessor, WidePreprocessor

if __name__ == "__main__":

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

    # useless assertion to avoid mypy warnings
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

    # binarize the ratings.
    df["rating"] = df["rating"].apply(lambda x: 1 if x >= 4 else 0)

    # sort by timestamp, groupby user and keep the one before the last for val
    # and the last for test
    df = df.sort_values(by=["timestamp"])
    train_df = df.groupby("user_id").apply(lambda x: x.iloc[:-2]).reset_index(drop=True)
    val_df = df.groupby("user_id").apply(lambda x: x.iloc[-2]).reset_index(drop=True)
    test_df = df.groupby("user_id").apply(lambda x: x.iloc[-1]).reset_index(drop=True)
    assert len(df) == len(train_df) + len(val_df) + len(test_df)

    cat_cols = [
        "user_id",
        "movie_id",
        "age",
        "gender",
        "occupation",
        "genre_list",
    ]

    tab_preprocessor = TabPreprocessor(cat_embed_cols=cat_cols, for_mf=True)
    X_tab_tr = tab_preprocessor.fit_transform(train_df)
    X_tab_val = tab_preprocessor.transform(val_df)
    X_tab_te = tab_preprocessor.transform(test_df)

    wide_preprocessor = WidePreprocessor(wide_cols=cat_cols)
    X_wide_tr = wide_preprocessor.fit_transform(train_df)
    X_wide_val = wide_preprocessor.transform(val_df)
    X_wide_te = wide_preprocessor.transform(test_df)

    cat_embed_input: List[Tuple[str, int]] = tab_preprocessor.cat_embed_input
    dfm = DeepFactorizationMachine(
        column_idx=tab_preprocessor.column_idx,
        num_factors=8,
        cat_embed_input=cat_embed_input,
        mlp_hidden_dims=[64, 32],
    )

    dffm = DeepFieldAwareFactorizationMachine(
        column_idx=tab_preprocessor.column_idx,
        num_factors=8,
        cat_embed_input=cat_embed_input,
        mlp_hidden_dims=[64, 32],
    )

    xdfm = ExtremeDeepFactorizationMachine(
        column_idx=tab_preprocessor.column_idx,
        input_dim=16,
        cat_embed_input=cat_embed_input,
        cin_layer_dims=[32, 16],
        mlp_hidden_dims=[64, 32],
    )

    wide = Wide(input_dim=X_tab_tr.max(), pred_dim=1)

    for fm_model in [dfm, dffm, xdfm]:

        fm_model = WideDeep(wide=wide, deeptabular=fm_model)

        trainer = Trainer(fm_model, objective="binary", metrics=[Accuracy()])

        X_train = {
            "X_wide": X_wide_tr,
            "X_tab": X_tab_tr,
            "target": train_df["rating"].values,
        }
        X_val = {
            "X_wide": X_wide_val,
            "X_tab": X_tab_val,
            "target": val_df["rating"].values,
        }
        X_test = {"X_wide": X_wide_te, "X_tab": X_tab_te}

        trainer.fit(X_train=X_train, X_val=X_val, n_epochs=2)
        trainer.predict(X_test={"X_wide": X_wide_te, "X_tab": X_tab_te})
