import pytest

from pytorch_widedeep.models import HFTokenizer

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
    tokenizer = HFTokenizer(model_name)
    X = tokenizer.encode(df.random_sentences.tolist())
    assert X.shape[0] == df.shape[0]
