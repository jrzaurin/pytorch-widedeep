import pytest

import numpy as np

from pytorch_widedeep.models import HFTokenizer
from pytorch_widedeep.utils.fastai_transforms import (
    fix_html,
    spec_add_spaces,
    rm_useless_spaces,
)

from .generate_fake_data import generate

df = generate()

model_names = [
    "distilbert-base-uncased",
    "bert-base-uncased",
    "FacebookAI/roberta-base",
    "albert-base-v2",
    "google/electra-base-discriminator",
]


@pytest.mark.parametrize("model_name", model_names)
def test_tokenizer_basic_usage(model_name):
    with pytest.warns(UserWarning):
        tokenizer = HFTokenizer(model_name)
        X = tokenizer.encode(df.random_sentences.tolist())
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
            model_name="distilbert-base-uncased", use_fast_tokenizer=False
        )
        X = tokenizer.encode(df.random_sentences.tolist())
    assert X.shape[0] == df.shape[0]


def test_tokenizer_multiprocessing():
    with pytest.warns(UserWarning):
        tokenizer = HFTokenizer(model_name="distilbert-base-uncased", num_workers=2)
        X = tokenizer.encode(df.random_sentences.tolist())
    assert X.shape[0] == df.shape[0]


def test_tokenizer_with_params():
    tokenizer = HFTokenizer(model_name="distilbert-base-uncased")
    X = tokenizer.encode(
        df.random_sentences.tolist(),
        max_length=15,
        padding="max_length",
        truncation=True,
        add_special_tokens=True,
    )

    special_tokens = np.array(list(tokenizer.tokenizer.added_tokens_decoder.keys()))

    for i in range(X.shape[0]):
        assert len(np.intersect1d(special_tokens, X[i])) > 0

    assert X.shape[0] == df.shape[0] and X.shape[1] == 15


def test_tokenizer_decode():
    # TO DO add some intersection assertion
    with pytest.warns(UserWarning):
        tokenizer = HFTokenizer(model_name="distilbert-base-uncased")
        X = tokenizer.encode(df.random_sentences.tolist())
        texts = tokenizer.decode(X, skip_special_tokens=True)
    assert len(texts) == df.shape[0]
    assert isinstance(texts[0], str)

