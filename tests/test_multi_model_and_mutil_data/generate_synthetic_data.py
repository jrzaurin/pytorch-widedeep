# sometimes I call this script generate_fake_data.py
import os
import random
from typing import Tuple
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from faker import Faker


def generate_fake_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    current_dir = os.path.dirname(os.path.realpath(__file__))

    # Set seed for reproducibility
    random.seed(42)

    Faker.seed(42)

    num_rows = 64 + 16 + 16

    # Generate random categorical data
    categories = ["category_A", "category_B", "category_C"]

    cat_col = [random.choice(categories) for _ in range(num_rows)]

    # Generate random numerical data
    num_col = [np.random.rand() for _ in range(num_rows)]

    # Generate random sentences
    fake = Faker()
    text_col1 = [fake.sentence() for _ in range(num_rows)]
    text_col2 = [fake.sentence() for _ in range(num_rows)]

    # Generate the image data
    img_folder = "images"

    img_path = "/".join([current_dir, "load_from_folder_test_data", img_folder])

    if not os.path.exists(img_path):
        os.makedirs(img_path)

    for i in range(num_rows):
        image = np.random.randint(0, 256, (16, 16, 3), dtype="uint8")
        image_name = "image_set1_{}.png".format(i)
        cv2.imwrite("/".join([img_path, image_name]), image)

        image = np.random.randint(0, 256, (16, 16, 3), dtype="uint8")
        image_name = "image_set2_{}.png".format(i)
        cv2.imwrite("/".join([img_path, image_name]), image)

    # Generate fake target values
    target = [random.choice([0, 1]) for _ in range(num_rows)]

    # Create DataFrame
    data = {
        "cat_col": cat_col,
        "num_col": num_col,
        "text_col1": text_col1,
        "text_col2": text_col2,
        "image_col1": ["image_set1_{}.png".format(i) for i in range(num_rows)],
        "image_col2": ["image_set2_{}.png".format(i) for i in range(num_rows)],
        "target": target,
    }

    df = pd.DataFrame(data)

    save_dir = Path(current_dir) / "load_from_folder_test_data"

    if not save_dir.exists():
        save_dir.mkdir(parents=True)

    train_df = df.iloc[:64]
    val_df = df.iloc[64:80]
    test_df = df.iloc[80:]

    train_df.to_csv(save_dir / "train.csv", index=False)
    val_df.to_csv(save_dir / "val.csv", index=False)
    test_df.to_csv(save_dir / "test.csv", index=False)

    print("Dataset and images created and saved successfully.")

    return train_df, val_df, test_df


def generate_fake_data_for_mutil_tabular_components() -> (
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
):

    current_dir = os.path.dirname(os.path.realpath(__file__))
    save_dir = Path(current_dir) / "data_for_muti_tabular_components"

    if not save_dir.exists():
        save_dir.mkdir(parents=True)

    fake = Faker()

    random.seed(42)
    np.random.seed(42)

    # Create User Features DataFrame
    user_ids = range(1, 33)
    ages = np.random.randint(18, 65, size=32)
    genders = np.random.choice(["male", "female"], size=32)
    locations = np.random.choice(["location_a", "location_b", "location_c"], size=32)
    reviews = [fake.sentence(nb_words=10) for _ in range(32)]

    user_features = pd.DataFrame(
        {
            "id": user_ids,
            "age": ages,
            "gender": genders,
            "location": locations,
            "review": reviews,
        }
    )

    # Create Item Features DataFrame
    item_ids = range(1, 33)
    prices = np.round(np.random.uniform(10, 1000, size=32), 2)
    colors = np.random.choice(["red", "blue", "green", "yellow"], size=32)
    categories = np.random.choice(["category_1", "category_2", "category_3"], size=32)
    descriptions = [fake.sentence(nb_words=10) for _ in range(32)]

    item_features = pd.DataFrame(
        {
            "id": item_ids,
            "price": prices,
            "color": colors,
            "category": categories,
            "description": descriptions,
        }
    )

    # Create Interaction DataFrame
    interaction_data = []
    for _ in range(1000):  # maybe 1000 interactions is too much for a test
        user_id = random.choice(user_ids)
        item_id = random.choice(item_ids)
        purchased = random.choice([0, 1])
        interaction_data.append([user_id, item_id, purchased])

    interactions = pd.DataFrame(
        interaction_data, columns=["user_id", "item_id", "purchased"]
    )

    user_item_purchased_df = interactions.merge(
        user_features, left_on="user_id", right_on="id"
    ).merge(item_features, left_on="item_id", right_on="id")

    train_df = user_item_purchased_df.iloc[:800]
    val_df = user_item_purchased_df.iloc[800:900]
    test_df = user_item_purchased_df.iloc[900:]

    train_df.to_csv(save_dir / "train.csv", index=False)
    val_df.to_csv(save_dir / "val.csv", index=False)
    test_df.to_csv(save_dir / "test.csv", index=False)

    return train_df, val_df, test_df


if __name__ == "__main__":
    # _, _, _ = generate_fake_data()
    _, _, _ = generate_fake_data_for_mutil_tabular_components()
