import re

import numpy as np
import torch
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from pytorch_widedeep.datasets import load_movielens100k
from pytorch_widedeep.preprocessing import TabPreprocessor
from pytorch_widedeep.models.rec.din import DeepInterestNetwork


def label_encode_column(df, column_name):
    le = LabelEncoder()
    df[column_name] = le.fit_transform(df[column_name])
    return df, le


def create_sequences(group):
    movies = group["movie_id"].tolist()
    genres = group["genre_list"].tolist()
    ratings = group["rating"].tolist()

    sequences = []
    for i in range(len(movies) - 5):
        user_movies_sequence = movies[i : i + 5]
        genres_sequence = genres[i : i + 5]
        ratings_sequence = ratings[i : i + 5]
        target_item = movies[i + 5]
        target_item_rating = ratings[i + 5]

        sequences.append(
            {
                "user_id": group.name,
                "user_movies_sequence": user_movies_sequence,
                "genres_sequence": genres_sequence,
                "ratings_sequence": ratings_sequence,
                "target_item": target_item,
                "target_item_rating": target_item_rating,
            }
        )

    return pd.DataFrame(sequences)


def preprocess_movie_data(df):
    df_sorted = df.sort_values(["user_id", "timestamp"])

    result_df = (
        df_sorted.groupby("user_id").apply(create_sequences).reset_index(drop=True)
    )

    return result_df


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

    def clean_genre_list(genre_list):
        return "_".join(
            sorted([re.sub(r"[^a-z0-9]", "", genre.lower()) for genre in genre_list])
        )

    items["genre_list"] = items["genre_list"].apply(clean_genre_list)

    df = pd.merge(data, items[["movie_id", "genre_list"]], on="movie_id")

    df["rating"] = df["rating"].apply(lambda x: 1 if x >= 4 else 0)

    df, user_le = label_encode_column(df, "movie_id")
    df, genre_le = label_encode_column(df, "genre_list")

    df["movie_id"] = df["movie_id"] + 1
    df["genre_list"] = df["genre_list"] + 1
    df["rating"] = df["rating"] + 1  # need to add 1 to account for padding

    seq_df = preprocess_movie_data(df)

    X_target = np.array(seq_df.target_item.tolist()).reshape(-1, 1)

    X_user_behaviour = np.array(
        [lst + [0] * (5 - len(lst)) for lst in seq_df.user_movies_sequence.tolist()]
    )
    X_ratings = np.array(
        [lst + [0] * (5 - len(lst)) for lst in seq_df.ratings_sequence.tolist()]
    )
    X_genres = np.array(
        [lst + [0] * (5 - len(lst)) for lst in seq_df.genres_sequence.tolist()]
    )

    df_other_feat = pd.merge(
        seq_df[["user_id"]],
        users[["user_id", "age", "gender", "occupation"]],
        on="user_id",
    )

    tab_preprocessor = TabPreprocessor(
        cat_embed_cols=["user_id", "age", "gender", "occupation"]
    )

    X_other_feats = tab_preprocessor.fit_transform(df_other_feat)

    X_all = np.concatenate(
        [X_other_feats, X_target, X_user_behaviour, X_ratings, X_genres], axis=1
    )

    user_behaviour_cols = [f"item_{i+1}" for i in range(5)]
    genres_cols = [f"genre_{i+1}" for i in range(5)]
    ratings_cols = [f"rating_{i+1}" for i in range(5)]

    all_cols = (
        [el[0] for el in tab_preprocessor.cat_embed_input]
        + ["target_item"]
        + user_behaviour_cols
        + ratings_cols
        + genres_cols
    )
    column_idx = {k: i for i, k in enumerate(all_cols)}

    user_behavior_confiq = (
        user_behaviour_cols,
        X_user_behaviour.max(),
        32,
    )

    rating_seq_config = (ratings_cols, 2, 1)

    other_seq_cols_confiq = [(genres_cols, X_genres.max(), 16)]

    other_cols_config = tab_preprocessor.cat_embed_input

    din = DeepInterestNetwork(
        column_idx=column_idx,
        target_item_col="target_item",
        user_behavior_confiq=user_behavior_confiq,
        rating_seq_config=rating_seq_config,
        other_seq_cols_confiq=other_seq_cols_confiq,
        other_cols_config=other_cols_config,
        mlp_hidden_dims=[200, 100],
    )

    X_all_sample = X_all[:32]

    X_all_sample_tsnr = torch.from_numpy(X_all_sample).float()

    out = din(X_all_sample_tsnr)
