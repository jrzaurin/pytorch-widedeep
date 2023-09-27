import os

import pandas as pd
from torch.utils.data import DataLoader

from pytorch_widedeep.models import (
    Wide,
    TabMlp,
    Vision,
    BasicRNN,
    WideDeep,
    TabTransformer,
)
from pytorch_widedeep.training import TrainerFromFolder
from pytorch_widedeep.callbacks import EarlyStopping
from pytorch_widedeep.preprocessing import (
    ImagePreprocessor,
    ChunkTabPreprocessor,
    ChunkTextPreprocessor,
    ChunkWidePreprocessor,
)
from pytorch_widedeep.load_from_folder import (
    TabFromFolder,
    TextFromFolder,
    WideFromFolder,
    ImageFromFolder,
    WideDeepDatasetFromFolder,
)

full_path = os.path.realpath(__file__)
path = os.path.split(full_path)[0]

data_folder = os.path.join(path, "load_from_folder_test_data")
img_folder = os.path.join(data_folder, "images")

fname = "synthetic_dataset.csv"
img_col = "images"
text_col = "text"
cat_cols = ["category1", "category2"]
num_cols = ["numeric1", "numeric2"]

data_size = 32
chunksize = 8
n_chunks = data_size // chunksize


def _build_preprocessors(tab_params={}):
    wide_preprocessor = ChunkWidePreprocessor(
        wide_cols=cat_cols,
        n_chunks=n_chunks,
    )

    tab_preprocessor = ChunkTabPreprocessor(
        embed_cols=cat_cols,
        continuous_cols=num_cols,
        n_chunks=n_chunks,
        default_embed_dim=8,
        verbose=0,
        **tab_params,
    )

    text_preprocessor = ChunkTextPreprocessor(
        n_chunks=n_chunks, text_col=text_col, n_cpus=1, max_vocab=50, maxlen=10
    )

    img_preprocessor = ImagePreprocessor(
        img_col=img_col,
        img_path=img_folder,
    )

    for i, chunk in enumerate(
        pd.read_csv("/".join([data_folder, fname]), chunksize=chunksize)
    ):
        # the image processor does not need to be fitted before passed to the
        # loaders from folder
        wide_preprocessor.fit(chunk)
        tab_preprocessor.fit(chunk)
        text_preprocessor.fit(chunk)

    return wide_preprocessor, tab_preprocessor, text_preprocessor, img_preprocessor


def _build_loaders_from_folder(
    wide_preprocessor,
    tab_preprocessor,
    text_preprocessor,
    img_preprocessor,
):
    tab_from_folder = TabFromFolder(
        fname=fname,
        directory=data_folder,
        target_col="target",
        preprocessor=tab_preprocessor,
        img_col=img_col,
        text_col=text_col,
    )

    wide_from_folder = WideFromFolder(
        fname=fname,
        directory=data_folder,
        preprocessor=wide_preprocessor,
        ignore_target=True,
    )

    text_from_folder = TextFromFolder(
        preprocessor=text_preprocessor,
    )

    img_from_folder = ImageFromFolder(preprocessor=img_preprocessor)

    return wide_from_folder, tab_from_folder, text_from_folder, img_from_folder


def _build_full_data_loader_from_folder(
    wide_from_folder, tab_from_folder, text_from_folder, img_from_folder
):
    dataset_from_folder = WideDeepDatasetFromFolder(
        n_samples=data_size,
        wide_from_folder=wide_from_folder,
        tab_from_folder=tab_from_folder,
        text_from_folder=text_from_folder,
        img_from_folder=img_from_folder,
    )

    dataloader_from_folder = DataLoader(dataset_from_folder, batch_size=4)

    return dataloader_from_folder


def _build_eval_and_test_loaders(
    wide_from_folder, tab_from_folder, eval_fname, test_fname
):
    eval_wide_from_folder = TabFromFolder(fname=eval_fname, reference=wide_from_folder)
    test_wide_from_folder = TabFromFolder(
        fname=test_fname, reference=wide_from_folder, ignore_target=True
    )

    eval_tab_from_folder = TabFromFolder(fname=eval_fname, reference=tab_from_folder)
    test_tab_from_folder = TabFromFolder(
        fname=test_fname, reference=tab_from_folder, ignore_target=True
    )

    return (
        eval_wide_from_folder,
        test_wide_from_folder,
        eval_tab_from_folder,
        test_tab_from_folder,
    )


def _buid_model(
    wide_preprocessor, tab_preprocessor, text_preprocessor, with_attention=False
):
    wide = Wide(input_dim=wide_preprocessor.wide_dim)

    if with_attention:
        deeptabular = TabTransformer(
            column_idx=tab_preprocessor.column_idx,
            cat_embed_input=tab_preprocessor.cat_embed_input,
            continuous_cols=tab_preprocessor.continuous_cols,
            input_dim=8,
            n_heads=2,
            n_blocks=2,
        )
    else:
        deeptabular = TabMlp(
            mlp_hidden_dims=[16, 8],
            column_idx=tab_preprocessor.column_idx,
            cat_embed_input=tab_preprocessor.cat_embed_input,
            continuous_cols=tab_preprocessor.continuous_cols,
        )

    basic_rnn = BasicRNN(
        vocab_size=len(text_preprocessor.vocab.itos),
        embed_dim=8,
        hidden_dim=8,
    )

    basic_cnn = Vision()

    model = WideDeep(
        wide=wide,
        deeptabular=deeptabular,
        deeptext=basic_rnn,
        deepimage=basic_cnn,
    )

    return model


def test_trainer_from_loader_basic_inputs():
    (
        wide_preprocessor,
        tab_preprocessor,
        text_preprocessor,
        img_preprocessor,
    ) = _build_preprocessors()

    (
        wide_from_folder,
        tab_from_folder,
        text_from_folder,
        img_from_folder,
    ) = _build_loaders_from_folder(
        wide_preprocessor, tab_preprocessor, text_preprocessor, img_preprocessor
    )

    dataloader_from_folder = _build_full_data_loader_from_folder(
        wide_from_folder, tab_from_folder, text_from_folder, img_from_folder
    )

    model = _buid_model(wide_preprocessor, tab_preprocessor, text_preprocessor)

    trainer = TrainerFromFolder(
        model,
        objective="regression",
        verbose=1,
    )

    trainer.fit(
        train_loader=dataloader_from_folder,
    )

    # assertion here (TBD)
    assert True
