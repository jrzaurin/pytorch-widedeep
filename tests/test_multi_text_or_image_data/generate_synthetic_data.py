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


if __name__ == "__main__":
    _, _, _ = generate_fake_data()
