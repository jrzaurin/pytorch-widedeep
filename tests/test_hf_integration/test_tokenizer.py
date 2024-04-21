import numpy as np
import pytest

from pytorch_widedeep.preprocessing import HFPreprocessor as HFTokenizer
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

encoder_params = {
    "max_length": 15,
    "padding": "max_length",
    "truncation": True,
    "add_special_tokens": True,
}


# TO DO: check the chunk preprocessor and load from folder with a HFPReprocessor


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
