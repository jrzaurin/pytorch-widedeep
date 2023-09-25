import os

import pandas as pd

from pytorch_widedeep.preprocessing import (
    TabPreprocessor,
    TextPreprocessor,
    ImagePreprocessor,
    ChunkTabPreprocessor,
    ChunkTextPreprocessor,
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


def test_chunk_tab_preprocessor():
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


def test_chunk_text_preprocessor():
    df = pd.read_csv(os.path.join(data_folder, fname))
    text_processor = TextPreprocessor(text_col=text_col, n_cpus=1, maxlen=10)
    X_text = text_processor.fit_transform(df)

    chunk_text_processor = ChunkTextPreprocessor(
        text_col=text_col, n_chunks=n_chunks, n_cpus=1, maxlen=10
    )
    for chunk in pd.read_csv(os.path.join(data_folder, fname), chunksize=chunksize):
        chunk_text_processor.partial_fit(chunk)
    X_text_chunk = chunk_text_processor.transform(df)

    reconstruced_df = text_processor.inverse_transform(X_text)
    reconstruced_df_chunk = chunk_text_processor.inverse_transform(X_text_chunk)

    assert reconstruced_df.equals(reconstruced_df_chunk)
