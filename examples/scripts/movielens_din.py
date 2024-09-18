# DIN is a "special" model and the data needs a very particular preparation
# process. Therefore, the library does not have a dedicated preprocessor for
# this specific model. Below is a detail explanation and if someone wants to
# use this algo I would suggest to wrap it all in an object prior to pass the
# data to the model

import re
from functools import partial

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from pytorch_widedeep import Trainer
from pytorch_widedeep.models import WideDeep
from pytorch_widedeep.metrics import Accuracy
from pytorch_widedeep.datasets import load_movielens100k
from pytorch_widedeep.preprocessing import TabPreprocessor
from pytorch_widedeep.models.rec.din import DeepInterestNetwork


def clean_genre_list(genre_list):
    return "_".join(
        sorted([re.sub(r"[^a-z0-9]", "", genre.lower()) for genre in genre_list])
    )


def label_encode_column(df, column_name):
    le = LabelEncoder()
    df[column_name] = le.fit_transform(df[column_name])
    return df, le


def create_sequences(group, seq_len=5):
    movies = group["movie_id"].tolist()
    genres = group["genre_list"].tolist()
    ratings = group["rating"].tolist()

    sequences = []
    for i in range(len(movies) - seq_len):
        user_movies_sequence = movies[i : i + seq_len]
        genres_sequence = genres[i : i + seq_len]
        ratings_sequence = ratings[i : i + seq_len]
        target_item = movies[i + seq_len]
        target_item_rating = ratings[i + seq_len]

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

    seq_df = pd.DataFrame(sequences)
    non_seq_cols = group.drop_duplicates(["user_id"]).drop(
        ["movie_id", "genre_list", "rating", "timestamp"], axis=1
    )

    return pd.merge(seq_df, non_seq_cols, on="user_id")


def preprocess_movie_data(df, seq_len=5):
    df_sorted = df.sort_values(["user_id", "timestamp"])

    partial_create_sequences = partial(create_sequences, seq_len=seq_len)

    result_df = (
        df_sorted.groupby("user_id")
        .apply(partial_create_sequences)
        .reset_index(drop=True)
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

    items["genre_list"] = items["genre_list"].apply(clean_genre_list)

    df = pd.merge(data, items[["movie_id", "genre_list"]], on="movie_id")
    df = pd.merge(
        df,
        users[["user_id", "age", "gender", "occupation"]],
        on="user_id",
    )

    df["rating"] = df["rating"].apply(lambda x: 1 if x >= 4 else 0)

    # Up until here, everything is quite standard. Now, some columns will be
    # treated as sequences, and therefore, they need to be tokenized
    # and "numericalised"/label encoded
    df, user_le = label_encode_column(df, "user_id")
    df, item_le = label_encode_column(df, "movie_id")
    df, genre_le = label_encode_column(df, "genre_list")

    # Internally all models for tabular data in this libray use padding idx =
    # 0 for unseen values, while sklearn's LabelEncoder starts at 0.
    # Therefore we need to add 1 to the encoded values to leave 0 for
    # unknown/unseen values
    df["movie_id"] = df["movie_id"] + 1
    df["genre_list"] = df["genre_list"] + 1

    # The explanation as to why we do this with the ratings will come later
    df["rating"] = df["rating"] + 1

    # we build sequences of 5 movies. Our goal will be predicting whether the
    # next movie will be reviewed positively or negatively
    df = df.sort_values(by=["timestamp"]).reset_index(drop=True)
    seq_df = preprocess_movie_data(df, seq_len=5)
    # target back to 0/1
    seq_df["target_item_rating"] = seq_df["target_item_rating"] - 1

    X_target_item = np.array(seq_df.target_item.tolist()).reshape(-1, 1)

    # in reality, all users here have more than 5 reviews, so we have complete
    # sequences, but there is a change that this does not happen in a given
    # dataset, so one would have to pad with padding idx (0 in this case)
    seq_len = 5
    X_user_behaviour = np.array(
        [
            lst + [0] * (seq_len - len(lst))
            for lst in seq_df.user_movies_sequence.tolist()
        ]
    )
    X_ratings = np.array(
        [lst + [0] * (seq_len - len(lst)) for lst in seq_df.ratings_sequence.tolist()]
    )
    X_genres = np.array(
        [lst + [0] * (seq_len - len(lst)) for lst in seq_df.genres_sequence.tolist()]
    )

    # At this point we have the target item as an array of shape (N obs, 1),
    # and all columns that are going to be treated as sequences stored in
    # arrays of shape (N obs, seq_len). The rest of the columns are going to
    # be treated as ANY other "standard" tabular dataset
    other_cols = ["user_id", "age", "gender", "occupation"]
    df_other_feat = seq_df[other_cols]
    tab_preprocessor = TabPreprocessor(cat_embed_cols=other_cols)
    X_other_feats = tab_preprocessor.fit_transform(df_other_feat)

    X_all = np.concatenate(
        [X_other_feats, X_target_item, X_user_behaviour, X_ratings, X_genres], axis=1
    )

    # Now, all the model components in this library they take a tensor
    # (just one) as input. Therefore, if we want to treat some data
    # differently, we need to slice the tensor internally. For this to happen
    # we need to "tell" the algorithm which column is which. DIN has data of
    # two different natures: sequences and standard tabular (i.e. everything
    # else). For the sequences, lets simply define columns with what they are
    # an an index (but you can define then with whichever string you want)
    user_behaviour_cols = [f"item_{i+1}" for i in range(5)]
    genres_cols = [f"genre_{i+1}" for i in range(5)]
    ratings_cols = [f"rating_{i+1}" for i in range(5)]

    # Then all columns in the datasets are, in order of appearance in X_all:
    all_cols = (
        [el[0] for el in tab_preprocessor.cat_embed_input]  # tabular cols
        + ["target_item"]  # target item
        + user_behaviour_cols  # user behaviour seq cols
        + ratings_cols  # ratings seq cols
        + genres_cols  # genres seq cols
    )
    column_idx = {k: i for i, k in enumerate(all_cols)}

    # Now we need to define the so called "configs". For the sequence columns these will consist of:
    # - the column names
    # - the maximum value in the column (to define the number of embeddings)
    # - the embedding dim
    user_behavior_confiq = (
        user_behaviour_cols,
        X_user_behaviour.max(),
        32,
    )

    # Again, the explanation to this will come when we instantiate the model
    rating_seq_config = (ratings_cols, 2, 1)

    # all the other sequence columns that are not user behaviour or an action
    # related to the items that define the user behaviour will be refer
    # as "other sequence columns" and will be pass as elements of a list
    other_seq_cols_confiq = [(genres_cols, X_genres.max(), 16)]

    # And finally, the config for the remaining, tabular columns
    other_cols_config = tab_preprocessor.cat_embed_input

    # Now, one of the params of the DeepInterestNetwork is action_seq_config.
    # This 'action' can be, for example a rating, or purchased/not-purchased.
    # The way that this 'action' will be used is the following: this action
    # will **always** be learned as a 1d embedding and will be combined with
    # the user behaviour. For example, imagine that the action is
    # purchased/not-purchased. then per item in the user behaviour sequence
    # there will be a binary action to learn 0/1. Such action will be
    # represented by a float number (so 3 floats will be learned, one for
    # purchased, one for not-purchased and one for padding) that will
    # multiply the corresponding item embedding in the user behaviour
    # sequence.
    din = DeepInterestNetwork(
        column_idx=column_idx,
        target_item_col="target_item",
        user_behavior_confiq=user_behavior_confiq,
        action_seq_config=rating_seq_config,
        other_seq_cols_confiq=other_seq_cols_confiq,
        cat_embed_input=other_cols_config,
        mlp_hidden_dims=[128, 64],
    )

    # And from here on, everything is standard
    model = WideDeep(deeptabular=din)

    trainer = Trainer(model=model, objective="binary", metrics=[Accuracy()])

    # in the real world you would have to split the data into train, val and test
    trainer.fit(
        X_tab=X_all,
        target=seq_df.target_item_rating.values,
        n_epochs=5,
        batch_size=512,
    )
