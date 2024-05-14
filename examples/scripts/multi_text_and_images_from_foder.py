import os

import cv2
import numpy as np
import torch
import pandas as pd
from faker import Faker
from torch.utils.data import DataLoader

from pytorch_widedeep.models import TabMlp, Vision, BasicRNN, WideDeep
from pytorch_widedeep.training import TrainerFromFolder

# from pytorch_widedeep.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_widedeep.preprocessing import (
    ImagePreprocessor,
    ChunkTabPreprocessor,
    ChunkTextPreprocessor,
)
from pytorch_widedeep.load_from_folder import (
    TabFromFolder,
    TextFromFolder,
    ImageFromFolder,
    WideDeepDatasetFromFolder,
)

use_cuda = torch.cuda.is_available()


fake = Faker()


def generate_fake_data():
    np.random.seed(0)
    categories = np.random.choice(["A", "B", "C"], size=300)
    numerical = np.random.rand(300)
    text1 = [fake.text() for _ in range(300)]
    text2 = [fake.text() for _ in range(300)]
    image_files = ["image_{}.png".format(i) for i in range(1, 301)]
    target = np.random.rand(300)

    data = {
        "cat_col": categories,
        "num_col": numerical,
        "text_col1": text1,
        "text_col2": text2,
        "image_col": image_files,
        "target": target,
    }

    df = pd.DataFrame(data)

    current_dir = os.path.dirname(os.path.realpath(__file__))
    folder_name = "/".join([current_dir, "fake_data"])

    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    train_df = df.iloc[:200]
    val_df = df.iloc[200:250]
    test_df = df.iloc[250:]

    train_df.to_csv("/".join([folder_name, "train.csv"]), index=False)
    val_df.to_csv("/".join([folder_name, "val.csv"]), index=False)
    test_df.to_csv("/".join([folder_name, "test.csv"]), index=False)

    # Generate and save random 32x32 images to a folder called fake_images
    img_folder_name = "fake_images"
    img_path = "/".join([folder_name, img_folder_name])

    if not os.path.exists(img_path):
        os.makedirs(img_path)

    for i in range(300):
        image = np.random.randint(
            0, 256, (32, 32, 3), dtype=np.uint8
        )  # generate random 32x32 image
        cv2.imwrite("/".join([img_path, "image_{}.png".format(i + 1)]), image)

    print("Dataset and images created and saved successfully.")


if __name__ == "__main__":

    current_dir = os.path.dirname(os.path.realpath(__file__))
    data_dir = "/".join([current_dir, "fake_data"])
    img_dir = "/".join([data_dir, "fake_images"])

    # if the train data and the images are not already created, generate them
    if not os.path.exists("fake_data"):
        generate_fake_data()
    else:
        print("Dataset and images already created")

    train_fname = "train.csv"
    eval_fname = "val.csv"
    test_fname = "test.csv"

    train_size = 200
    val_size = 50
    test_size = 50

    chunksize = 50
    n_chunks = int(np.ceil(train_size / chunksize))

    img_col = "image_col"
    text_cols = ["text_col1", "text_col2"]
    cat_embed_cols = ["cat_col"]
    cont_cols = ["num_col"]
    target_col = "target"

    tab_preprocessor = ChunkTabPreprocessor(
        embed_cols=cat_embed_cols,
        continuous_cols=cont_cols,
        n_chunks=n_chunks,
        default_embed_dim=4,
        verbose=0,
    )

    text_preprocessor_1 = ChunkTextPreprocessor(
        n_chunks=n_chunks, text_col=text_cols[0], n_cpus=1, maxlen=20
    )

    text_preprocessor_2 = ChunkTextPreprocessor(
        n_chunks=n_chunks,
        text_col=text_cols[1],
        n_cpus=1,
        maxlen=20,
    )

    img_preprocessor = ImagePreprocessor(
        img_col=img_col,
        img_path=img_dir,
    )

    for i, chunk in enumerate(
        pd.read_csv("/".join([data_dir, train_fname]), chunksize=chunksize)
    ):
        print(f"chunk in loop: {i}")
        tab_preprocessor.fit(chunk)
        text_preprocessor_1.fit(chunk)
        text_preprocessor_2.fit(chunk)

    train_tab_folder = TabFromFolder(
        fname="dataset.csv",
        directory=data_dir,
        target_col=target_col,
        preprocessor=tab_preprocessor,
        text_col=text_cols,
        img_col=img_col,
    )
    eval_tab_folder = TabFromFolder(fname=eval_fname, reference=train_tab_folder)  # type: ignore[arg-type]
    test_tab_folder = TabFromFolder(
        fname=test_fname, reference=train_tab_folder, ignore_target=True  # type: ignore[arg-type]
    )

    text_folder = TextFromFolder(
        preprocessor=[text_preprocessor_1, text_preprocessor_2],
    )

    img_folder = ImageFromFolder(preprocessor=img_preprocessor)

    train_dataset_folder = WideDeepDatasetFromFolder(
        n_samples=train_size,
        tab_from_folder=train_tab_folder,
        text_from_folder=text_folder,
        img_from_folder=img_folder,
    )
    eval_dataset_folder = WideDeepDatasetFromFolder(
        n_samples=val_size,
        tab_from_folder=eval_tab_folder,
        reference=train_dataset_folder,
    )
    test_dataset_folder = WideDeepDatasetFromFolder(
        n_samples=test_size,
        tab_from_folder=test_tab_folder,
        reference=train_dataset_folder,
    )
    train_loader = DataLoader(train_dataset_folder, batch_size=16, num_workers=1)
    eval_loader = DataLoader(eval_dataset_folder, batch_size=16, num_workers=1)
    test_loader = DataLoader(test_dataset_folder, batch_size=16, num_workers=1)

    basic_rnn_1 = BasicRNN(
        vocab_size=len(text_preprocessor_1.vocab.itos),
        n_layers=1,
        embed_dim=16,
        hidden_dim=32,
    )

    basic_rnn_2 = BasicRNN(
        vocab_size=len(text_preprocessor_2.vocab.itos),
        n_layers=1,
        embed_dim=8,
        hidden_dim=16,
    )

    deepimage = Vision(
        channel_sizes=[32, 64],
        kernel_sizes=[3, 3],
        strides=[1, 1],
        head_hidden_dims=[16, 8],
    )

    deepdense = TabMlp(
        mlp_hidden_dims=[64, 32],
        column_idx=tab_preprocessor.column_idx,
        cat_embed_input=tab_preprocessor.cat_embed_input,
        continuous_cols=cont_cols,
    )

    model = WideDeep(
        deeptabular=deepdense,
        deeptext=[basic_rnn_1, basic_rnn_2],
        deepimage=deepimage,
    )

    trainer = TrainerFromFolder(model, objective="regression")

    trainer.fit(train_loader=train_loader)
