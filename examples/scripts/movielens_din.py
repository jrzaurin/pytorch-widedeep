import re

import pandas as pd

from pytorch_widedeep import Trainer
from pytorch_widedeep.models import WideDeep
from pytorch_widedeep.metrics import Accuracy
from pytorch_widedeep.datasets import load_movielens100k
from pytorch_widedeep.preprocessing import DINPreprocessor
from pytorch_widedeep.models.rec.din import DeepInterestNetwork


def clean_genre_list(genre_list):
    return "_".join(
        sorted([re.sub(r"[^a-z0-9]", "", genre.lower()) for genre in genre_list])
    )


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

    assert (
        isinstance(items, pd.DataFrame)
        and isinstance(data, pd.DataFrame)
        and isinstance(users, pd.DataFrame)
    )
    items["genre_list"] = items[list_of_genres].apply(
        lambda x: [genre for genre in list_of_genres if x[genre] == 1], axis=1
    )

    items["genre_list"] = items["genre_list"].apply(clean_genre_list)

    df = pd.merge(data, items[["movie_id", "genre_list"]], on="movie_id")
    df = pd.merge(
        df,
        users[["user_id", "age", "gender", "occupation"]],
        on="user_id",
    )

    df["rating"] = df["rating"].apply(lambda x: 1 if x >= 4 else 0)

    din_preprocessor = DINPreprocessor(
        user_id_col="user_id",
        item_embed_col="movie_id",
        target_col="rating",
        max_seq_length=5,
        action_col="rating",
        other_seq_embed_cols=["genre_list"],
        cat_embed_cols=["user_id", "age", "gender", "occupation"],
    )

    X_din, y = din_preprocessor.fit_transform(df)

    din = DeepInterestNetwork(
        column_idx=din_preprocessor.din_columns_idx,
        target_item_col="target_item",
        user_behavior_confiq=din_preprocessor.user_behaviour_config,
        action_seq_config=din_preprocessor.action_seq_config,
        other_seq_cols_confiq=din_preprocessor.other_seq_config,
        cat_embed_input=din_preprocessor.tab_preprocessor.cat_embed_input,
        mlp_hidden_dims=[128, 64],
    )

    # And from here on, everything is standard
    model = WideDeep(deeptabular=din)

    trainer = Trainer(model=model, objective="binary", metrics=[Accuracy()])

    # in the real world you would have to split the data into train, val and test
    trainer.fit(
        X_tab=X_din,
        target=y,
        n_epochs=5,
        batch_size=512,
    )
