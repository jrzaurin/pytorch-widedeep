import numpy as np
import pandas as pd
import pytest
import warnings

from sklearn.datasets import fetch_20newsgroups
from pytorch_widedeep.preprocessing import TextProcessor

texts = np.random.choice(fetch_20newsgroups().data, 10)
df = pd.DataFrame({'texts':texts})
processor = TextProcessor(min_freq=0)
X_text = processor.fit_transform(df, 'texts')

###############################################################################
# There is not much to test here. I will simply test that the tokenization and
# and padding processes went well
###############################################################################
def test_text_processor():
	idx = int(np.random.choice(np.arange(10), 1))

	original_tokens= processor.tokens[idx]
	if len(original_tokens) > processor.maxlen:
		original_tokens = original_tokens[-processor.maxlen:]

	padded_sequence = X_text[idx]

	recovered_tokens=[]
	for t in padded_sequence:
		if 	processor.vocab.itos[t] != 'xxpad':
			recovered_tokens.append(processor.vocab.itos[t])

	assert np.all([org == recv for org, recv in zip(original_tokens, recovered_tokens)])
