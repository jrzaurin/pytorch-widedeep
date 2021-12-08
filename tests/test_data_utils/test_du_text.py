import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import fetch_20newsgroups
from sklearn.exceptions import NotFittedError

from pytorch_widedeep.utils import text_utils
from pytorch_widedeep.preprocessing import TextPreprocessor

texts = np.random.choice(fetch_20newsgroups().data, 10)
df = pd.DataFrame({"texts": texts})
processor = TextPreprocessor(min_freq=0, text_col="texts")
X_text = processor.fit_transform(df)


###############################################################################
# There is not much to test here. I will simply test that the tokenization and
# and padding processes went well
###############################################################################
def test_text_processor():
    idx = int(np.random.choice(np.arange(10), 1))

    original_tokens = processor.tokens[idx]
    if len(original_tokens) > processor.maxlen:
        original_tokens = original_tokens[-processor.maxlen :]

    padded_sequence = X_text[idx]

    recovered_tokens = []
    for t in padded_sequence:
        if processor.vocab.itos[t] != "xxpad":
            recovered_tokens.append(processor.vocab.itos[t])

    assert np.all([org == recv for org, recv in zip(original_tokens, recovered_tokens)])


def test_pad_sequences():
    out = []
    seq = [1, 2, 3]
    padded_seq_1 = text_utils.pad_sequences(seq, maxlen=5, pad_idx=0)
    out.append(all([el == 0 for el in padded_seq_1[:2]]))
    padded_seq_2 = text_utils.pad_sequences(seq, maxlen=5, pad_idx=1, pad_first=False)
    out.append(all([el == 1 for el in padded_seq_2[-2:]]))
    assert all(out)


###############################################################################
# Test inverse transform
###############################################################################
def test_inverse_transform():

    df = pd.DataFrame(
        {
            "text_column": [
                "life is like a box of chocolates",
                "You never know what you're going to get",
            ]
        }
    )

    text_preprocessor = TextPreprocessor(
        text_col="text_column", max_vocab=25, min_freq=1, maxlen=10, verbose=False
    )
    padded_seq = text_preprocessor.fit_transform(df)
    org_df = text_preprocessor.inverse_transform(padded_seq)

    texts = org_df.text_column.values

    assert ("life is like box of chocolates" in texts[0]) and (
        "you never know what you re going to get" in texts[1]
    )


###############################################################################
# Test NotFittedError
###############################################################################


def test_notfittederror():
    processor = TextPreprocessor(min_freq=0, text_col="texts")
    with pytest.raises(NotFittedError):
        processor.transform(df)
