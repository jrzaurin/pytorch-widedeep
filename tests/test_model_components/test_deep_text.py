import numpy as np
import torch
import pytest

from pytorch_widedeep.models import DeepText

padded_sequences = np.random.choice(np.arange(1,100), (100, 48))
padded_sequences = np.hstack((np.repeat(np.array([[0,0]]), 100, axis=0), padded_sequences))
pretrained_embeddings = np.random.rand(1000, 64)
vocab_size = 1000

###############################################################################
# Without Pretrained Embeddings
###############################################################################
model1 = DeepText(
    vocab_size=vocab_size,
    embed_dim=32,
    padding_idx=0
    )
def test_deep_test():
	out = model1(torch.from_numpy(padded_sequences))
	assert out.size(0)==100 and out.size(1)==1

###############################################################################
# With Pretrained Embeddings
###############################################################################
model2 = DeepText(
    vocab_size=vocab_size,
    embedding_matrix=pretrained_embeddings,
    padding_idx=0
    )
def test_deep_test_pretrained():
	out = model2(torch.from_numpy(padded_sequences))
	assert out.size(0)==100 and out.size(1)==1

###############################################################################
# Make sure it throws a warning
###############################################################################
def test_catch_warning():
	with pytest.warns(UserWarning):
		model3 = DeepText(
		    vocab_size=vocab_size,
		    embed_dim=32,
		    embedding_matrix=pretrained_embeddings,
		    padding_idx=0
		    )
