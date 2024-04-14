import random

import pandas as pd
from faker import Faker


def generate(seed: int = 42) -> pd.DataFrame:
    # thank ChatGPT for this code

    # Set seed for reproducibility
    random.seed(seed)

    # Instantiate Faker
    Faker.seed(seed)
    fake = Faker()

    # Define the number of rows
    num_rows = 32

    # Generate random categorical data
    categories = ["category_A", "category_B", "category_C"]
    cat_col1 = [random.choice(categories) for _ in range(num_rows)]
    cat_col2 = [random.choice(categories) for _ in range(num_rows)]

    # Generate random numerical data
    num_col1 = [random.randint(1, 100) for _ in range(num_rows)]
    num_col2 = [random.randint(1, 100) for _ in range(num_rows)]

    # Generate random sentences
    sentences = [fake.sentence() for _ in range(num_rows)]

    # Generate fake target values
    targets = [random.choice([0, 1]) for _ in range(num_rows)]

    # Create DataFrame
    df = pd.DataFrame(
        {
            "cat_col1": cat_col1,
            "cat_col2": cat_col2,
            "num_col1": num_col1,
            "num_col2": num_col2,
            "random_sentences": sentences,
            "target": targets,
        }
    )

    return df
