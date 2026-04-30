import re

import pandas as pd

from pytorch_widedeep import Trainer
from pytorch_widedeep.models import WideDeep
from pytorch_widedeep.metrics import Accuracy
from pytorch_widedeep.datasets import load_movielens100k
from pytorch_widedeep.preprocessing import TabPreprocessor
from pytorch_widedeep.models.rec.rankmixer import RankMixer


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

    df["user_activity"] = df.groupby("user_id")["rating"].transform("count")
    df["item_popularity"] = df.groupby("movie_id")["rating"].transform("count")

    cat_embed_cols = [
        ("user_id", 8),
        ("gender", 8),
        ("occupation", 8),
        ("movie_id", 8),
        ("genre_list", 8),
    ]
    continuous_cols = ["age", "user_activity", "item_popularity"]

    tab_preprocessor = TabPreprocessor(
        cat_embed_cols=cat_embed_cols,
        continuous_cols=continuous_cols,
        cols_to_scale=["age", "user_activity", "item_popularity"],
    )

    X_tab = tab_preprocessor.fit_transform(df)
    y = df["rating"].values

    column_groups = [
        ["user_id", "gender", "user_activity", "occupation", "age"],
        ["movie_id", "genre_list", "item_popularity"],
    ]

    rankmixer = RankMixer(
        column_idx=tab_preprocessor.column_idx,
        column_groups=column_groups,
        cat_embed_input=tab_preprocessor.cat_embed_input,
        continuous_cols=tab_preprocessor.continuous_cols,
        embed_continuous_method="standard",
        cont_embed_dim=8,
        num_tokens=4,
        token_size=16,
        model_dim=64,
        num_layers=2,
        num_heads=4,
        ff_factor=4.0,
        ff_dropout=0.1,
        use_moe=True,
        num_experts=8,
        top_k=2,
    )

    # And from here on, everything is standard
    model = WideDeep(deeptabular=rankmixer)

    trainer = Trainer(model=model, objective="binary", metrics=[Accuracy()])

    # in the real world you would have to split the data into train, val and test
    trainer.fit(
        X_tab=X_tab,
        target=y,
        n_epochs=5,
        batch_size=512,
    )
