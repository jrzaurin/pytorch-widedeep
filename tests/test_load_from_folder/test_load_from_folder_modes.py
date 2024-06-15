import os

import torch
import pandas as pd
import pytest
from torchvision import transforms

from pytorch_widedeep.preprocessing import (
    ImagePreprocessor,
    ChunkHFPreprocessor,
    ChunkTabPreprocessor,
    ChunkTextPreprocessor,
    ChunkWidePreprocessor,
)
from pytorch_widedeep.load_from_folder import (
    TabFromFolder,
    TextFromFolder,
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


def test_tab_from_folder_alone():
    df = pd.read_csv("/".join([data_folder, fname]))

    tab_preprocessor = ChunkTabPreprocessor(
        embed_cols=cat_cols,
        continuous_cols=num_cols,
        n_chunks=n_chunks,
    )

    for i, chunk in enumerate(
        pd.read_csv("/".join([data_folder, fname]), chunksize=chunksize)
    ):
        tab_preprocessor.fit(chunk)

    tab_from_folder = TabFromFolder(
        fname=fname,
        directory=data_folder,
        target_col="target_regression",
        preprocessor=tab_preprocessor,
    )

    processed_sample = tab_preprocessor.transform(df)[1]
    processed_sample_from_folder, _, _, _ = tab_from_folder.get_item(1)

    assert (processed_sample == processed_sample_from_folder).all()


def test_tab_from_folder_with_reference():
    df = pd.read_csv("/".join([data_folder, fname]))

    tab_preprocessor = ChunkTabPreprocessor(
        embed_cols=cat_cols,
        continuous_cols=num_cols,
        n_chunks=n_chunks,
    )

    for i, chunk in enumerate(
        pd.read_csv("/".join([data_folder, fname]), chunksize=chunksize)
    ):
        tab_preprocessor.fit(chunk)

    train_tab_from_folder = TabFromFolder(
        fname=fname,
        directory=data_folder,
        target_col="target_regression",
        preprocessor=tab_preprocessor,
    )

    eval_tab_from_folder = TabFromFolder(
        fname=fname,  # this should really be the eval_fname, but for tests purposes is fine
        reference=train_tab_from_folder,
    )

    processed_sample = tab_preprocessor.transform(df)[1]
    processed_sample_from_folder, _, _, _ = eval_tab_from_folder.get_item(1)

    assert (processed_sample == processed_sample_from_folder).all()


def test_text_from_folder_alone():
    df = pd.read_csv("/".join([data_folder, fname]))

    chunk_text_processor = ChunkTextPreprocessor(
        text_col=text_col, n_chunks=1, n_cpus=1, maxlen=10, max_vocab=50
    )
    for chunk in pd.read_csv("/".join([data_folder, fname]), chunksize=chunksize):
        chunk_text_processor.partial_fit(chunk)

    text_folder = TextFromFolder(preprocessor=chunk_text_processor)

    processed_sample = chunk_text_processor.transform(df)[1]
    processed_sample_from_folder = text_folder.get_item(df.text.loc[1])

    assert (processed_sample == processed_sample_from_folder).all()


def test_image_from_folder_alone():
    df = pd.read_csv("/".join([data_folder, fname]))

    img_preprocessor = ImagePreprocessor(
        img_col=img_col,
        img_path=img_folder,
    )

    img_from_folder = ImageFromFolder(preprocessor=img_preprocessor)

    processed_sample = img_preprocessor.transform(df)[1]
    processed_sample = processed_sample.transpose(2, 0, 1)

    processed_sample_from_folder = img_from_folder.get_item(df.images.loc[1])

    # we can only assert that the images have the same shape
    assert processed_sample.shape == processed_sample_from_folder.shape


def test_image_from_folder_with_transforms():
    df = pd.read_csv("/".join([data_folder, fname]))

    img_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.CenterCrop(10),
        ]
    )
    img_from_folder = ImageFromFolder(directory=img_folder, transforms=img_transforms)

    processed_sample_from_folder = img_from_folder.get_item(df.images.loc[1])

    return processed_sample_from_folder.shape == torch.Size([3, 10, 10])


@pytest.mark.parametrize("hugginface", [True, False])
def test_full_wide_deep_dataset_from_folder(hugginface):
    df = pd.read_csv("/".join([data_folder, fname]))

    tab_preprocessor = ChunkTabPreprocessor(
        embed_cols=cat_cols,
        continuous_cols=num_cols,
        n_chunks=n_chunks,
        default_embed_dim=8,
        verbose=0,
    )

    if not hugginface:
        text_preprocessor = ChunkTextPreprocessor(
            n_chunks=n_chunks,
            text_col=text_col,
            n_cpus=1,
            maxlen=10,
            max_vocab=50,
        )
    else:
        text_preprocessor = ChunkHFPreprocessor(
            model_name="distilbert-base-uncased",
            text_col=text_col,
            encode_params={
                "max_length": 20,
                "padding": "max_length",
                "truncation": True,
            },
        )

    img_preprocessor = ImagePreprocessor(
        img_col=img_col,
        img_path=img_folder,
    )

    for i, chunk in enumerate(
        pd.read_csv("/".join([data_folder, fname]), chunksize=chunksize)
    ):
        tab_preprocessor.fit(chunk)
        # if it is ChunkHFPreprocessor, the fit method does nothing
        text_preprocessor.fit(chunk)

    tab_from_folder = TabFromFolder(
        fname=fname,
        directory=data_folder,
        target_col="target_regression",
        preprocessor=tab_preprocessor,
        text_col=text_col,
        img_col=img_col,
    )

    text_from_folder = TextFromFolder(
        preprocessor=text_preprocessor,
    )

    img_from_folder = ImageFromFolder(preprocessor=img_preprocessor)

    train_dataset_folder = WideDeepDatasetFromFolder(
        n_samples=df.shape[0],
        tab_from_folder=tab_from_folder,
        text_from_folder=text_from_folder,
        img_from_folder=img_from_folder,
    )

    X, y = train_dataset_folder.__getitem__(1)

    cond1 = all([k in X for k in ["deeptabular", "deeptext", "deepimage"]])
    cond2 = X["deeptabular"].shape[0] == len(cat_cols) + len(num_cols)
    cond3 = X["deeptext"].shape[0] == (20 if hugginface else 10)
    cond4 = X["deepimage"].shape == (3, 224, 224)

    assert all([cond1, cond2, cond3, cond4])


@pytest.mark.parametrize("tabular_component", ["wide", "deeptabular"])
def test_wide_and_tab_optional(tabular_component):
    df = pd.read_csv("/".join([data_folder, fname]))

    if tabular_component == "wide":
        tab_preprocessor = ChunkWidePreprocessor(
            wide_cols=cat_cols,
            n_chunks=n_chunks,
        )
    else:
        tab_preprocessor = ChunkTabPreprocessor(
            embed_cols=cat_cols,
            continuous_cols=num_cols,
            n_chunks=n_chunks,
            default_embed_dim=8,
            verbose=0,
        )

    text_preprocessor = ChunkTextPreprocessor(
        n_chunks=n_chunks,
        text_col=text_col,
        n_cpus=1,
        maxlen=10,
        max_vocab=50,
    )

    img_preprocessor = ImagePreprocessor(
        img_col=img_col,
        img_path=img_folder,
    )

    for i, chunk in enumerate(
        pd.read_csv("/".join([data_folder, fname]), chunksize=chunksize)
    ):
        tab_preprocessor.fit(chunk)
        text_preprocessor.fit(chunk)

    tab_from_folder = TabFromFolder(
        fname=fname,
        directory=data_folder,
        target_col="target_regression",
        preprocessor=tab_preprocessor,
        text_col=text_col,
        img_col=img_col,
    )

    text_from_folder = TextFromFolder(
        preprocessor=text_preprocessor,
    )

    img_from_folder = ImageFromFolder(preprocessor=img_preprocessor)

    train_dataset_folder = WideDeepDatasetFromFolder(
        n_samples=df.shape[0],
        tab_from_folder=tab_from_folder if tabular_component == "deeptabular" else None,
        wide_from_folder=tab_from_folder if tabular_component == "wide" else None,
        text_from_folder=text_from_folder,
        img_from_folder=img_from_folder,
    )

    X, y = train_dataset_folder.__getitem__(1)

    if tabular_component == "deeptabular":
        cond1 = all([k in X for k in ["deeptabular", "deeptext", "deepimage"]])
        cond2 = X["deeptabular"].shape[0] == len(cat_cols) + len(num_cols)
    else:
        cond1 = all([k in X for k in ["wide", "deeptext", "deepimage"]])
        cond2 = X["wide"].shape[0] == len(cat_cols)

    cond3 = X["deeptext"].shape[0] == text_preprocessor.maxlen
    cond4 = X["deepimage"].shape == (3, 224, 224)

    assert all([cond1, cond2, cond3, cond4])
