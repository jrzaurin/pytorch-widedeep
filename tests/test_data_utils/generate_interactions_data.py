import os
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

full_path = os.path.realpath(__file__)
path = os.path.split(full_path)[0]

save_dir = Path(path) / "interactions_df_for_rec_preprocessor"


def generate_sample_data(n_users=5, n_items=10, seed=42, return_df=False):
    """
    Generate a sample dataset for recommendation system testing.

    Parameters:
    -----------
    - n_users: int, number of users to generate (default 5)
    - n_items: int, number of items to generate (default 10)
    - seed: int, random seed for reproducibility (default 42)
    """
    np.random.seed(seed)

    # Generate user data
    users = pd.DataFrame(
        {
            "user_id": range(1, n_users + 1),
            "age": np.random.randint(18, 65, n_users),
            "gender": np.random.choice(["M", "F"], n_users),
            "height": np.random.uniform(150, 200, n_users).round(1),
            "weight": np.random.uniform(50, 100, n_users).round(1),
        }
    )

    # Generate item data
    items = pd.DataFrame(
        {
            "item_id": range(1, n_items + 1),
            "price": np.random.uniform(10, 100, n_items).round(2),
            "category": np.random.choice(["A", "B", "C"], n_items),
        }
    )

    # Generate positive interactions
    positive_interactions = []
    start_date = datetime(2023, 1, 1)

    for user_id in users["user_id"]:
        # Randomly select 5 items for this user to interact with
        user_items = np.random.choice(items["item_id"], 5, replace=False)

        if user_id == 1:
            n_interactions = 3
        else:
            n_interactions = np.random.randint(4, 21)

        for _ in range(n_interactions):
            item_id = np.random.choice(user_items)
            timestamp = start_date + timedelta(days=np.random.randint(0, 365))
            positive_interactions.append(
                {
                    "user_id": user_id,
                    "item_id": item_id,
                    "timestamp": timestamp,
                    "interaction": 1,  # Positive interaction
                }
            )

    positive_df = pd.DataFrame(positive_interactions)

    # Generate negative interactions
    negative_interactions = []
    for user_id in users["user_id"]:
        positive_items = positive_df[positive_df["user_id"] == user_id][
            "item_id"
        ].unique()
        negative_items = items[~items["item_id"].isin(positive_items)]["item_id"]

        n_negative = len(positive_df[positive_df["user_id"] == user_id])

        for _ in range(n_negative):
            item_id = np.random.choice(negative_items)
            timestamp = start_date + timedelta(days=np.random.randint(0, 365))
            negative_interactions.append(
                {
                    "user_id": user_id,
                    "item_id": item_id,
                    "timestamp": timestamp,
                    "interaction": 0,  # Negative interaction
                }
            )

    negative_df = pd.DataFrame(negative_interactions)

    # Combine positive and negative interactions
    interactions_df = pd.concat([positive_df, negative_df], ignore_index=True)

    # Merge all data
    final_df = interactions_df.merge(users, on="user_id").merge(items, on="item_id")

    # Sort by timestamp
    final_df = final_df.sort_values("timestamp").reset_index(drop=True)

    # Reorder columns
    column_order = [
        "user_id",
        "item_id",
        "age",
        "gender",
        "height",
        "weight",
        "price",
        "category",
        "timestamp",
        "interaction",
    ]
    final_df = final_df[column_order]

    if not save_dir.exists():
        save_dir.mkdir(parents=True)

    if not return_df:
        final_df.to_csv(save_dir / "interactions_data.csv", index=False)
    else:
        return final_df


def split_by_timestamp(df, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """
    Split the dataframe into train, validation, and test sets based on timestamp.

    Parameters:
    - df: pd.DataFrame, the input dataframe
    - train_ratio: float, ratio of data for training (default 0.8)
    - val_ratio: float, ratio of data for validation (default 0.1)
    - test_ratio: float, ratio of data for testing (default 0.1)

    Returns:
    - train_df, val_df, test_df: pd.DataFrame, the split dataframes
    """
    assert np.isclose(train_ratio + val_ratio + test_ratio, 1.0), "Ratios must sum to 1"

    df_sorted = df.sort_values("timestamp")
    total_rows = len(df_sorted)
    train_rows = int(total_rows * train_ratio)
    val_rows = int(total_rows * val_ratio)

    train_df = df_sorted.iloc[:train_rows]
    val_df = df_sorted.iloc[train_rows : train_rows + val_rows]
    test_df = df_sorted.iloc[train_rows + val_rows :]

    return train_df, val_df, test_df


def split_by_last_interactions(df):
    """
    Split the dataframe into train, validation, and test sets based on the last interactions.
    Train: all interactions but last two
    Validation: second to last interaction
    Test: last interaction

    Parameters:
    - df: pd.DataFrame, the input dataframe

    Returns:
    - train_df, val_df, test_df: pd.DataFrame, the split dataframes
    """
    df_sorted = df.sort_values(["user_id", "timestamp"])

    # Get the last two interactions for each user
    last_two = df_sorted.groupby("user_id").tail(2)

    # Split the last two interactions into validation and test
    test_df = last_two.groupby("user_id").last().reset_index()
    val_df = last_two.groupby("user_id").first().reset_index()

    # Remove the last two interactions from the original dataframe to create the train set
    train_df = df_sorted[~df_sorted.index.isin(last_two.index)].reset_index(drop=True)

    return train_df, val_df, test_df


if __name__ == "__main__":
    generate_sample_data()

    # df = generate_sample_data(return_df=True)

    # # Split the data by timestamp
    # train_df, val_df, test_df = split_by_timestamp(df)

    # # Split the data by last interactions
    # train_df2, val_df2, test_df2 = split_by_last_interactions(df)
