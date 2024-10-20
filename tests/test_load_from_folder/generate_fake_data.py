# Script Almost fully generated by copilot :)
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

full_path = os.path.realpath(__file__)
path = os.path.split(full_path)[0]

save_dir = Path(path) / "load_from_folder_test_data"

if not save_dir.exists():
    np.random.seed(42)

    # Sample sentences
    sample_sentences = [
        "The quick brown fox jumps over the lazy dog.",
        "A journey of a thousand miles begins with a single step.",
        "All that glitters is not gold.",
        "Actions speak louder than words.",
        "Beauty is in the eye of the beholder.",
        "Don't count your chickens before they hatch.",
        "Every cloud has a silver lining.",
        "When in Rome, do as the Romans do.",
    ]

    save_dir.mkdir(parents=True)

    images_save_dir = save_dir / "images"
    images_save_dir.mkdir(parents=True)

    # Create a list to store the data
    data = []

    # Generate 32 observations
    for i in range(32):
        # Generate categorical data
        cat1 = random.choice(["A", "B", "C"])
        cat2 = random.choice(["X", "Y"])

        # Generate numerical data
        num1 = random.uniform(1, 100)
        num2 = random.uniform(1, 100)

        # Select a random sentence from the sample sentences
        text = random.choice(sample_sentences)

        # Generate random noise images
        # Append all data for the current row, including the image names list
        data.append([cat1, cat2, num1, num2, text])

    noise_images = [
        Image.fromarray(np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8))
        for _ in range(32)
    ]

    # Generate image names and save images to the "images" subfolder
    image_names = []
    for j, img in enumerate(noise_images):
        image_name = f"image_{j}.png"
        img.save(images_save_dir / image_name, format="PNG")
        image_names.append(image_name)

    # Create a DataFrame
    columns = ["category1", "category2", "numeric1", "numeric2", "text"]
    df = pd.DataFrame(data, columns=columns)
    df["images"] = image_names

    df["target_multiclass"] = np.random.randint(0, 3, 32)
    df["target_binary"] = np.random.randint(0, 2, 32)
    df["target_regression"] = np.random.rand(32)

    # Save the DataFrame to a CSV file in the specified directory
    df.to_csv(save_dir / "synthetic_dataset.csv", index=False)
