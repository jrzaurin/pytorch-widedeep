import os

import pandas as pd
import pytest

from pytorch_widedeep.preprocessing import (
    TabPreprocessor,
    TextPreprocessor,
    WidePreprocessor,
    ChunkTabPreprocessor,
    ChunkTextPreprocessor,
    ChunkWidePreprocessor,
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


def test_chunk_wide_processor_one_chunk():
    df = pd.read_csv(os.path.join(data_folder, fname))
    wide_processor = WidePreprocessor(wide_cols=cat_cols)
    X_wide = wide_processor.fit_transform(df)

    chunk_wide_processor = ChunkWidePreprocessor(wide_cols=cat_cols, n_chunks=1)
    chunk_wide_processor.partial_fit(df)

    X_wide_chunk = chunk_wide_processor.transform(df)

    reconstruced_df = wide_processor.inverse_transform(X_wide)
    reconstruced_df_chunk = chunk_wide_processor.inverse_transform(X_wide_chunk)

    assert reconstruced_df.equals(reconstruced_df_chunk)


def test_chunk_wide_processor():
    df = pd.read_csv(os.path.join(data_folder, fname))
    wide_processor = WidePreprocessor(wide_cols=cat_cols)
    X_wide = wide_processor.fit_transform(df)

    chunk_wide_processor = ChunkWidePreprocessor(wide_cols=cat_cols, n_chunks=n_chunks)
    for chunk in pd.read_csv(os.path.join(data_folder, fname), chunksize=chunksize):
        chunk_wide_processor.partial_fit(chunk)

    X_wide_chunk = chunk_wide_processor.transform(df)

    reconstruced_df = wide_processor.inverse_transform(X_wide)
    reconstruced_df_chunk = chunk_wide_processor.inverse_transform(X_wide_chunk)

    assert reconstruced_df.equals(reconstruced_df_chunk)


def test_chunk_tab_preprocessor_one_chunk():
    df = pd.read_csv(os.path.join(data_folder, fname))
    tab_processor = TabPreprocessor(cat_embed_cols=cat_cols, continuous_cols=num_cols)
    X_tab = tab_processor.fit_transform(df)

    chunk_tab_processor = ChunkTabPreprocessor(
        cat_embed_cols=cat_cols, continuous_cols=num_cols, n_chunks=1
    )
    chunk_tab_processor.partial_fit(df)
    X_tab_chunk = chunk_tab_processor.transform(df)

    assert (X_tab == X_tab_chunk).all()


def test_chunk_tab_preprocessor_without_params():
    df = pd.read_csv(os.path.join(data_folder, fname))
    tab_processor = TabPreprocessor(cat_embed_cols=cat_cols, continuous_cols=num_cols)
    X_tab = tab_processor.fit_transform(df)

    chunk_tab_processor = ChunkTabPreprocessor(
        cat_embed_cols=cat_cols, continuous_cols=num_cols, n_chunks=n_chunks
    )
    for chunk in pd.read_csv(os.path.join(data_folder, fname), chunksize=chunksize):
        chunk_tab_processor.partial_fit(chunk)
    X_tab_chunk = chunk_tab_processor.transform(df)

    reconstruced_df = tab_processor.inverse_transform(X_tab)
    reconstruced_df_chunk = chunk_tab_processor.inverse_transform(X_tab_chunk)

    assert reconstruced_df.equals(reconstruced_df_chunk)


@pytest.mark.parametrize("with_attention", [True, False])
@pytest.mark.parametrize("quantization_setup", [{"numeric2": [0.0, 50.0, 100.0]}, None])
def test_chunk_tab_preprocessor_with_params(with_attention, quantization_setup):
    df = pd.read_csv(os.path.join(data_folder, fname))
    tab_processor = TabPreprocessor(
        cat_embed_cols=cat_cols,
        continuous_cols=num_cols,
        cols_to_scale=["numeric1"],
        with_attention=with_attention,
        with_cls_token=with_attention,
        quantization_setup=quantization_setup,
    )
    X_tab = tab_processor.fit_transform(df)

    chunk_tab_processor = ChunkTabPreprocessor(
        n_chunks=n_chunks,
        cat_embed_cols=cat_cols,
        continuous_cols=num_cols,
        cols_to_scale=["numeric1"],
        with_attention=with_attention,
        with_cls_token=with_attention,
        cols_and_bins=quantization_setup,
    )
    for chunk in pd.read_csv(os.path.join(data_folder, fname), chunksize=chunksize):
        chunk_tab_processor.partial_fit(chunk)

    X_tab_chunk = chunk_tab_processor.transform(df)

    reconstruced_df_chunk = chunk_tab_processor.inverse_transform(X_tab_chunk)
    reconstruced_df = tab_processor.inverse_transform(X_tab)

    assert reconstruced_df.equals(reconstruced_df_chunk)


def test_chunk_text_preprocessor_one_go():
    df = pd.read_csv(os.path.join(data_folder, fname))
    text_processor = TextPreprocessor(
        text_col=text_col, n_cpus=1, maxlen=10, max_vocab=50
    )
    X_text = text_processor.fit_transform(df)

    chunk_text_processor = ChunkTextPreprocessor(
        text_col=text_col, n_chunks=1, n_cpus=1, maxlen=10, max_vocab=50
    )
    chunk_text_processor.partial_fit(df)
    X_text_chunk = chunk_text_processor.transform(df)

    assert (X_text == X_text_chunk).all()


def test_chunk_text_preprocessor():
    df = pd.read_csv(os.path.join(data_folder, fname))
    text_processor = TextPreprocessor(
        text_col=text_col, n_cpus=1, maxlen=10, max_vocab=50
    )
    X_text = text_processor.fit_transform(df)

    chunk_text_processor = ChunkTextPreprocessor(
        text_col=text_col, n_chunks=n_chunks, n_cpus=1, maxlen=10, max_vocab=50
    )
    for chunk in pd.read_csv(os.path.join(data_folder, fname), chunksize=chunksize):
        chunk_text_processor.partial_fit(chunk)
    X_text_chunk = chunk_text_processor.transform(df)

    reconstruced_df = text_processor.inverse_transform(X_text)
    reconstruced_df_chunk = chunk_text_processor.inverse_transform(X_text_chunk)

    assert reconstruced_df.equals(reconstruced_df_chunk)
