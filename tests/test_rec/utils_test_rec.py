import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from pytorch_widedeep.preprocessing import TabPreprocessor


def create_purchase_dataframe():
    np.random.seed(42)
    n_rows = 100

    df = pd.DataFrame(
        {
            "user_id": np.random.choice(range(1, 6), n_rows),
            "item_id": np.random.choice(range(1, 6), n_rows),
            "item_price": np.random.uniform(10, 100, n_rows).round(2),
            "item_category": np.random.choice(
                ["Electronics", "Clothing", "Books", "Home", "Sports"], n_rows
            ),
            "user_age": np.random.randint(18, 65, n_rows),
            "user_location": np.random.choice(
                ["New York", "Los Angeles", "Chicago", "Houston", "Phoenix"], n_rows
            ),
            "purchased": np.random.choice([0, 1], n_rows, p=[0.7, 0.3]),
        }
    )

    return df


def create_user_sequences(df, sequence_length=5):
    sequences = []

    for user_id in df["user_id"].unique():
        user_df = df[df["user_id"] == user_id].sort_index()

        if len(user_df) < sequence_length + 1:
            continue

        for i in range(len(user_df) - sequence_length):
            seq = user_df.iloc[i : i + sequence_length]
            target = user_df.iloc[i + sequence_length]

            sequences.append(
                {
                    "user_id": user_id,
                    "user_age": seq["user_age"].iloc[0],
                    "user_location": seq["user_location"].iloc[0],
                    "item_seq": seq["item_id"].tolist(),
                    "item_price_seq": seq["item_price"].tolist(),
                    "item_category_seq": seq["item_category"].tolist(),
                    "item_purchased_seq": seq["purchased"].tolist(),
                    "target_item": target["item_id"],
                    "purchased": target["purchased"],
                }
            )

    return pd.DataFrame(sequences)


def split_user_data(df):
    train_data = []
    val_data = []
    test_data = []

    for user_id in df["user_id"].unique():
        user_df = df[df["user_id"] == user_id].sort_index()
        n_interactions = len(user_df)

        if n_interactions >= 3:
            train_data.append(user_df.iloc[:-2])
            val_data.append(user_df.iloc[-2:-1])
            test_data.append(user_df.iloc[-1:])
        elif n_interactions == 2:
            train_data.append(user_df.iloc[0:1])
            val_data.append(user_df.iloc[1:2])
        elif n_interactions == 1:
            train_data.append(user_df)

    train_df = pd.concat(train_data, ignore_index=True)
    val_df = pd.concat(val_data, ignore_index=True)
    test_df = pd.concat(test_data, ignore_index=True)

    return train_df, val_df, test_df


def create_train_val_test_data():
    df = create_purchase_dataframe()
    train_df, val_df, test_df = split_user_data(df)
    return train_df, val_df, test_df


def label_encode_column(df, column_name):
    le = LabelEncoder()
    df[column_name] = le.fit_transform(df[column_name])
    return df, le


def prepare_data_for_din():
    df = create_and_encode_dataframe()
    df_seq = create_user_sequences(df)
    df_seq = process_item_seq_and_target(df_seq)
    train, val, test = split_user_data(df_seq)
    tab_preprocessor = TabPreprocessor(
        cat_embed_cols=["user_id", "user_age", "user_location"],
        continuous_cols=[f"item_price_{i}" for i in range(5)],
    )
    data = process_datasets(train, val, test, tab_preprocessor)
    config, column_idx = create_config(train, tab_preprocessor)
    return data, config, column_idx


def create_and_encode_dataframe():
    df = create_purchase_dataframe()
    df_le = df.copy()

    for column in ["item_id", "item_category", "user_location"]:
        df_le, _ = label_encode_column(df_le, column)
        df_le[column] += 1  # Add 1 because 0 will be used for padding

    df_le["purchased"] += 1
    return df_le


def process_item_seq_and_target(df_seq):
    df_seq["purchased"] = (df_seq["purchased"] - 1).astype(int)

    for i in range(5):
        df_seq[f"item_price_{i}"] = df_seq["item_price_seq"].apply(lambda x: x[i])

    df_seq.drop(columns="item_price_seq", inplace=True)
    return df_seq


def process_dataset(dataset, tab_preprocessor, is_train=False):
    X_item_seq = np.array(dataset["item_seq"].tolist())
    X_item_purchased_seq = np.array(dataset["item_purchased_seq"].tolist())
    X_item_category_seq = np.array(dataset["item_category_seq"].tolist())
    X_target_item = dataset["target_item"].values.reshape(-1, 1)

    if is_train:
        X_other_cols = tab_preprocessor.fit_transform(dataset)
    else:
        X_other_cols = tab_preprocessor.transform(dataset)

    y = dataset["purchased"].values

    X = np.concatenate(
        [
            X_item_seq,
            X_item_purchased_seq,
            X_item_category_seq,
            X_other_cols,
            X_target_item,
        ],
        axis=1,
    )

    return X, y


def process_datasets(train, val, test, tab_preprocessor):
    return {
        "train": process_dataset(train, tab_preprocessor, is_train=True),
        "val": process_dataset(val, tab_preprocessor),
        "test": process_dataset(test, tab_preprocessor),
    }


def create_config(train, tab_preprocessor):
    item_seq_cols = [f"item_seq_{i}" for i in range(5)]
    item_purchased_seq_cols = [f"item_purchased_seq_{i}" for i in range(5)]
    item_category_seq_cols = [f"item_category_seq_{i}" for i in range(5)]

    col_order = (
        item_seq_cols
        + item_purchased_seq_cols
        + item_category_seq_cols
        + [el[0] for el in tab_preprocessor.cat_embed_input]
        + [f"item_price_{i}" for i in range(5)]
        + ["target_item"]
    )

    column_idx = {k: i for i, k in enumerate(col_order)}

    config = {
        "column_idx": column_idx,
        "item_seq_config": (
            item_seq_cols,
            np.array(train["item_seq"].tolist()).max(),
            8,
        ),
        "item_purchased_seq_config": (item_purchased_seq_cols, 2),
        "item_category_seq_config": [
            (
                item_category_seq_cols,
                np.array(train["item_category_seq"].tolist()).max(),
                8,
            )
        ],
        "cat_embed_input": tab_preprocessor.cat_embed_input,
        "continuous_cols": tab_preprocessor.continuous_cols,
    }

    return config, column_idx
