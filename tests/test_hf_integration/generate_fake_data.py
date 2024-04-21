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
    cat1 = [random.choice(categories) for _ in range(num_rows)]
    cat2 = [random.choice(categories) for _ in range(num_rows)]

    # Generate random numerical data
    num1 = [random.randint(1, 100) for _ in range(num_rows)]
    num2 = [random.randint(1, 100) for _ in range(num_rows)]

    # Generate random sentences
    sentences = [fake.sentence() for _ in range(num_rows)]

    # Generate fake target values
    targets = [random.choice([0, 1]) for _ in range(num_rows)]

    # Create DataFrame
    df = pd.DataFrame(
        {
            "cat1": cat1,
            "cat2": cat2,
            "num1": num1,
            "num2": num2,
            "random_sentences": sentences,
            "target": targets,
        }
    )

    return df
