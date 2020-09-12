import numpy as np
import pandas as pd
from sklearn.datasets import fetch_20newsgroups

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
