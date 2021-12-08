import numpy as np
import torch
import pytest

from pytorch_widedeep.models import AttentiveRNN

padded_sequences = np.random.choice(np.arange(1, 100), (100, 48))
padded_sequences = np.hstack(
    (np.repeat(np.array([[0, 0]]), 100, axis=0), padded_sequences)
)
pretrained_embeddings = np.random.rand(1000, 64).astype("float32")
vocab_size = 1000


# ###############################################################################
# # Test Basic Model without attention
# ###############################################################################


def test_basic_model():
    model = AttentiveRNN(vocab_size=vocab_size, embed_dim=32, padding_idx=0)
    out = model(torch.from_numpy(padded_sequences))
    assert out.size(0) == 100 and out.size(1) == 64


###############################################################################
# Without Pretrained Embeddings and attention
###############################################################################


@pytest.mark.parametrize(
    "bidirectional",
    [True, False],
)
@pytest.mark.parametrize(
    "attn_concatenate",
    [True, False],
)
def test_basic_model_with_attn(bidirectional, attn_concatenate):
    model = AttentiveRNN(
        vocab_size=vocab_size,
        embed_dim=32,
        padding_idx=0,
        hidden_dim=32,
        bidirectional=bidirectional,
        with_attention=True,
        attn_concatenate=attn_concatenate,
    )
    out = model(torch.from_numpy(padded_sequences))
    if attn_concatenate and bidirectional:
        out_size_1_ok = out.size(1) == model.embed_dim * 4
    elif attn_concatenate or bidirectional:
        out_size_1_ok = out.size(1) == model.embed_dim * 2
    else:
        out_size_1_ok = out.size(1) == model.embed_dim

    attn_weights_ok = model.attn.attn_weights.size(1) == padded_sequences.shape[1]

    assert out.size(0) == 100 and out_size_1_ok and attn_weights_ok


###############################################################################
# With Pretrained Embeddings
###############################################################################


def test_model_with_pretrained():
    model = AttentiveRNN(
        vocab_size=vocab_size, embed_matrix=pretrained_embeddings, padding_idx=0
    )
    out = model(torch.from_numpy(padded_sequences))
    assert out.size(0) == 100 and out.size(1) == 64


###############################################################################
# Make sure it throws a UserWarning when the input embedding dimension and the
# dimension of the pretrained embeddings do not match.
###############################################################################


def test_catch_warning():
    with pytest.warns(UserWarning):
        model = AttentiveRNN(
            vocab_size=vocab_size,
            embed_dim=32,
            embed_matrix=pretrained_embeddings,
            padding_idx=0,
        )
    out = model(torch.from_numpy(padded_sequences))
    assert out.size(0) == 100 and out.size(1) == 64


###############################################################################
# Without Pretrained Embeddings and head layers
###############################################################################


def test_model_with_head_layers():
    model = AttentiveRNN(
        vocab_size=vocab_size, embed_dim=32, padding_idx=0, head_hidden_dims=[64, 16]
    )
    out = model(torch.from_numpy(padded_sequences))
    assert out.size(0) == 100 and out.size(1) == 16


###############################################################################
# Pretrained Embeddings made non-trainable
###############################################################################


def test_embed_non_trainable():
    model = AttentiveRNN(
        vocab_size=vocab_size,
        embed_matrix=pretrained_embeddings,
        embed_trainable=False,
        padding_idx=0,
    )
    out = model(torch.from_numpy(padded_sequences))  # noqa: F841
    assert np.allclose(model.word_embed.weight.numpy(), pretrained_embeddings)


# ##############################################################################
# GRU and using output
# ##############################################################################


@pytest.mark.parametrize(
    "bidirectional",
    [True, False],
)
def test_gru_and_using_ouput(bidirectional):
    model = AttentiveRNN(
        vocab_size=vocab_size,
        rnn_type="gru",
        embed_dim=32,
        bidirectional=bidirectional,
        padding_idx=0,
        use_hidden_state=False,
    )
    out = model(torch.from_numpy(padded_sequences))  # noqa: F841
    assert out.size(0) == 100 and out.size(1) == model.output_dim
