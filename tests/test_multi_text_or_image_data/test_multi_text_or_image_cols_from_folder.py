import os

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from pytorch_widedeep.models import TabMlp, Vision, BasicRNN, WideDeep
from pytorch_widedeep.training import TrainerFromFolder
from pytorch_widedeep.preprocessing import (
    ChunkTabPreprocessor,
    ChunkTextPreprocessor,
)
from pytorch_widedeep.load_from_folder import (
    TabFromFolder,
    TextFromFolder,
    ImageFromFolder,
    WideDeepDatasetFromFolder,
)

current_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = "/".join([current_dir, "load_from_folder_test_data"])
img_dir = "/".join([data_dir, "images"])


train_fname = "train.csv"
eval_fname = "val.csv"
test_fname = "test.csv"

train_size = 64
val_size = 16
test_size = 16

chunksize = 16
n_chunks = int(np.ceil(train_size / chunksize))

text_cols = ["text_col1", "text_col2"]
img_cols = ["image_col1", "image_col2"]
cat_embed_cols = ["cat_col"]
cont_cols = ["num_col"]
target_col = "target"


def test_multi_text_or_image_cols_with_load_from_folder():
    # most of the functionalities here have already been tested elsewhere,
    # mainl in the test_load_from_folder dir. Here I am simply testing that
    # all runs without errors when using multiple text and image columns

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

    for i, chunk in enumerate(
        pd.read_csv("/".join([data_dir, train_fname]), chunksize=chunksize)
    ):
        tab_preprocessor.fit(chunk)
        text_preprocessor_1.fit(chunk)
        text_preprocessor_2.fit(chunk)

    train_tab_folder = TabFromFolder(
        fname="train.csv",
        directory=data_dir,
        target_col=target_col,
        preprocessor=tab_preprocessor,
        text_col=text_cols,
        img_col=img_cols,
    )
    eval_tab_folder = TabFromFolder(fname=eval_fname, reference=train_tab_folder)  # type: ignore[arg-type]
    test_tab_folder = TabFromFolder(
        fname=test_fname, reference=train_tab_folder, ignore_target=True  # type: ignore[arg-type]
    )

    text_from_folder = TextFromFolder(
        preprocessor=[text_preprocessor_1, text_preprocessor_2]
    )

    img_from_folder = ImageFromFolder(directory=img_dir)

    train_dataset_folder = WideDeepDatasetFromFolder(
        n_samples=train_size,
        tab_from_folder=train_tab_folder,
        text_from_folder=text_from_folder,
        img_from_folder=img_from_folder,
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

    tab_mlp = TabMlp(
        column_idx=tab_preprocessor.column_idx,
        cat_embed_input=tab_preprocessor.cat_embed_input,  # type: ignore[arg-type]
        continuous_cols=tab_preprocessor.continuous_cols,
        mlp_hidden_dims=[16, 4],
    )

    vision_1 = Vision(
        channel_sizes=[16, 32],
        kernel_sizes=[3, 3],
        strides=[1, 1],
        head_hidden_dims=[16, 8],
    )

    vision_2 = Vision(
        channel_sizes=[16, 32],
        kernel_sizes=[3, 3],
        strides=[1, 1],
        head_hidden_dims=[16, 4],
    )

    rnn_1 = BasicRNN(
        vocab_size=len(text_preprocessor_1.vocab.itos),
        embed_dim=16,
        hidden_dim=16,
        n_layers=1,
        bidirectional=False,
        head_hidden_dims=[16, 8],
    )

    rnn_2 = BasicRNN(
        vocab_size=len(text_preprocessor_2.vocab.itos),
        embed_dim=16,
        hidden_dim=16,
        n_layers=1,
        bidirectional=False,
        head_hidden_dims=[16, 4],
    )

    model = WideDeep(
        deeptabular=tab_mlp,
        deeptext=[rnn_1, rnn_2],
        deepimage=[vision_1, vision_2],
        pred_dim=1,
    )

    trainer = TrainerFromFolder(model, objective="binary")

    trainer.fit(train_loader=train_loader, eval_loader=eval_loader, n_epochs=1)

    preds = trainer.predict(test_loader=test_loader)

    assert trainer.history["train_loss"] is not None and len(preds) == 16
