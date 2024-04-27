import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pytorch_widedeep.preprocessing import HFPreprocessor as HFTokenizer
from pytorch_widedeep.preprocessing import ChunkHFPreprocessor
from pytorch_widedeep.load_from_folder import TextFromFolder
from pytorch_widedeep.utils.fastai_transforms import (
    fix_html,
    spec_add_spaces,
    rm_useless_spaces,
)

full_path = os.path.realpath(__file__)
path = Path(os.path.split(full_path)[0])

df = pd.read_csv(os.path.join(path, "load_from_folder_test_data", "data.csv"))

model_names = [
    "distilbert-base-uncased",
    "bert-base-uncased",
    "FacebookAI/roberta-base",
    "albert-base-v2",
    "google/electra-base-discriminator",
]

encoder_params = {
    "max_length": 15,
    "padding": "max_length",
    "truncation": True,
    "add_special_tokens": True,
}


@pytest.mark.parametrize("model_name", model_names)
def test_tokenizer_basic_usage(model_name):
    tokenizer = HFTokenizer(text_col="random_sentences", model_name=model_name)
    X = tokenizer.fit_transform(df)
    assert X.shape[0] == df.shape[0]


def test_tokenizer_preprocessing_rules():
    def to_lower(text):
        return text.lower()

    with pytest.warns(UserWarning):
        tokenizer = HFTokenizer(
            model_name="distilbert-base-uncased",
            preprocessing_rules=[
                to_lower,
                fix_html,
                spec_add_spaces,
                rm_useless_spaces,
            ],
        )
        X = tokenizer.encode(df.random_sentences.tolist())
    assert X.shape[0] == df.shape[0]


def test_tokenizer_use_fast_tokenizer():
    with pytest.warns(UserWarning):
        tokenizer = HFTokenizer(
            model_name="distilbert-base-uncased",
            use_fast_tokenizer=True,
        )
        X = tokenizer.encode(df.random_sentences.tolist())
    assert X.shape[0] == df.shape[0]


def test_tokenizer_multiprocessing():
    with pytest.warns(UserWarning):
        tokenizer = HFTokenizer(model_name="distilbert-base-uncased", num_workers=2)
        X = tokenizer.encode(df.random_sentences.tolist())
    assert X.shape[0] == df.shape[0]


def test_tokenizer_with_params():
    tokenizer = HFTokenizer(
        model_name="distilbert-base-uncased", encode_params=encoder_params
    )
    X = tokenizer.encode(
        df.random_sentences.tolist(),
    )

    special_tokens = np.array(list(tokenizer.tokenizer.added_tokens_decoder.keys()))

    for i in range(X.shape[0]):
        assert len(np.intersect1d(special_tokens, X[i])) > 0

    assert X.shape[0] == df.shape[0] and X.shape[1] == 15


def test_tokenizer_decode():
    with pytest.warns(UserWarning):
        tokenizer = HFTokenizer(model_name="distilbert-base-uncased")
        X = tokenizer.encode(df.random_sentences.tolist())
        texts = tokenizer.decode(X, skip_special_tokens=True)

    # This first assertion could be a bit more restrictive
    assert (
        len(
            set(df.random_sentences[0].lower().split()).intersection(
                set(texts[0].split())
            )
        )
        > 0
    )
    assert len(texts) == df.shape[0]
    assert isinstance(texts[0], str)


def test_text_from_folder_with_hftokenizer():
    # The HF ChunkPreprocessor is mostly the same as the HFPreprocessor. No
    # need for specific tests
    hf_preprocessor = HFTokenizer(
        model_name="distilbert-base-uncased",
    )
    X = hf_preprocessor.encode(df.random_sentences.tolist())

    chunk_hf_preprocessor = ChunkHFPreprocessor(
        model_name="distilbert-base-uncased",
        text_col="random_sentences",
        encode_params={
            "max_length": X.shape[1],
            "padding": "max_length",
            "truncation": True,
        },
    )

    text_folder = TextFromFolder(preprocessor=chunk_hf_preprocessor)
    processed_sample_from_folder = text_folder.get_item(df.random_sentences.loc[1])

    assert all(processed_sample_from_folder == X[1])


def test_load_text_files_from_folder():
    hf_preprocessor = HFTokenizer(
        model_name="distilbert-base-uncased",
        text_col="text_fnames",
        root_dir=os.path.join(path, "load_from_folder_test_data", "sentences"),
    )

    X = hf_preprocessor.fit_transform(df)

    assert X.shape[0] == df.shape[0]
