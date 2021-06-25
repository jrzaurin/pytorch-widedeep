import numpy as np
import torch
import pytest

from pytorch_widedeep.models import DeepText

padded_sequences = np.random.choice(np.arange(1, 100), (100, 48))
padded_sequences = np.hstack(
    (np.repeat(np.array([[0, 0]]), 100, axis=0), padded_sequences)
)
pretrained_embeddings = np.random.rand(1000, 64).astype("float32")
vocab_size = 1000


###############################################################################
# Without Pretrained Embeddings
###############################################################################
model1 = DeepText(vocab_size=vocab_size, embed_dim=32, padding_idx=0)


def test_deep_text():
    out = model1(torch.from_numpy(padded_sequences))
    assert out.size(0) == 100 and out.size(1) == 64


###############################################################################
# With Pretrained Embeddings
###############################################################################
model2 = DeepText(
    vocab_size=vocab_size, embed_matrix=pretrained_embeddings, padding_idx=0
)


def test_deep_text_pretrained():
    out = model2(torch.from_numpy(padded_sequences))
    assert out.size(0) == 100 and out.size(1) == 64


###############################################################################
# Make sure it throws a UserWarning when the input embedding dimension and the
# dimension of the pretrained embeddings do not match.
###############################################################################
def test_catch_warning():
    with pytest.warns(UserWarning):
        model3 = DeepText(
            vocab_size=vocab_size,
            embed_dim=32,
            embed_matrix=pretrained_embeddings,
            padding_idx=0,
        )
    out = model3(torch.from_numpy(padded_sequences))
    assert out.size(0) == 100 and out.size(1) == 64


###############################################################################
# Without Pretrained Embeddings and head layers
###############################################################################

model4 = DeepText(
    vocab_size=vocab_size, embed_dim=32, padding_idx=0, head_hidden_dims=[64, 16]
)


def test_deep_text_head_layers():
    out = model4(torch.from_numpy(padded_sequences))
    assert out.size(0) == 100 and out.size(1) == 16


###############################################################################
# Without Pretrained Embeddings, bidirectional
###############################################################################

model5 = DeepText(
    vocab_size=vocab_size, embed_dim=32, padding_idx=0, bidirectional=True
)


def test_deep_text_bidirectional():
    out = model1(torch.from_numpy(padded_sequences))
    assert out.size(0) == 100 and out.size(1) == 64


###############################################################################
# Pretrained Embeddings made non-trainable
###############################################################################

model6 = DeepText(
    vocab_size=vocab_size,
    embed_matrix=pretrained_embeddings,
    embed_trainable=False,
    padding_idx=0,
)


def test_embed_non_trainable():
    out = model6(torch.from_numpy(padded_sequences))  # noqa: F841
    assert np.allclose(model6.word_embed.weight.numpy(), pretrained_embeddings)


###############################################################################
# GRU and using output
###############################################################################

model7 = DeepText(
    vocab_size=vocab_size,
    rnn_type="gru",
    embed_dim=32,
    padding_idx=0,
    use_hidden_state=False,
)

model8 = DeepText(
    vocab_size=vocab_size,
    rnn_type="gru",
    embed_dim=32,
    padding_idx=0,
    bidirectional=True,
    use_hidden_state=False,
)


def test_gru_and_using_ouput():
    out = model7(torch.from_numpy(padded_sequences))  # noqa: F841
    out_bi = model7(torch.from_numpy(padded_sequences))  # noqa: F841
    assert out.size(0) == 100 and out.size(1) == 64 and out_bi.size(1) == 64
