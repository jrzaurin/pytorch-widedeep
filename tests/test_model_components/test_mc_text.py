import numpy as np
import torch
import pytest

from pytorch_widedeep.models import BasicRNN, AttentiveRNN, StackedAttentiveRNN

padded_sequences = np.random.choice(np.arange(1, 100), (100, 48))
padded_sequences = np.hstack(
    (np.repeat(np.array([[0, 0]]), 100, axis=0), padded_sequences)
)
pretrained_embeddings = np.random.rand(1000, 64).astype("float32")
vocab_size = 1000


# ###############################################################################
# # Test Basic Model with/without attention
# ###############################################################################


@pytest.mark.parametrize(
    "attention",
    [True, False],
)
def test_basic_model(attention):
    if not attention:
        model = BasicRNN(vocab_size=vocab_size, embed_dim=32, padding_idx=0)
    else:
        model = AttentiveRNN(vocab_size=vocab_size, embed_dim=32, padding_idx=0)
    out = model(torch.from_numpy(padded_sequences))

    res = []
    res.append(out.size(0) == 100)

    try:
        if model.attn_concatenate:
            res.append(out.size(1) == model.hidden_dim * 2)
    except Exception:
        res.append(out.size(1) == model.hidden_dim)

    assert all(res)


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
        attn_concatenate=attn_concatenate,
    )
    out = model(torch.from_numpy(padded_sequences))
    if attn_concatenate and bidirectional:
        out_size_1_ok = out.size(1) == model.hidden_dim * 4
    elif attn_concatenate or bidirectional:
        out_size_1_ok = out.size(1) == model.hidden_dim * 2
    else:
        out_size_1_ok = out.size(1) == model.hidden_dim

    attn_weights_ok = model.attn.attn_weights.size(1) == padded_sequences.shape[1]

    assert out.size(0) == 100 and out_size_1_ok and attn_weights_ok


###############################################################################
# With Pretrained Embeddings
###############################################################################


@pytest.mark.parametrize(
    "bidirectional",
    [True, False],
)
def test_model_with_pretrained(bidirectional):
    model = BasicRNN(
        vocab_size=vocab_size,
        embed_matrix=pretrained_embeddings,
        padding_idx=0,
        bidirectional=bidirectional,
    )
    out = model(torch.from_numpy(padded_sequences))
    assert (
        out.size(0) == 100 and out.size(1) == model.hidden_dim * 2
        if bidirectional
        else model.hidden_dim
    )


###############################################################################
# Make sure it throws a UserWarning when the input embedding dimension and the
# dimension of the pretrained embeddings do not match.
###############################################################################


@pytest.mark.parametrize(
    "model_name",
    ["basic", "stacked"],
)
def test_catch_warning(model_name):
    with pytest.warns(UserWarning):
        if model_name == "basic":
            model = BasicRNN(
                vocab_size=vocab_size,
                embed_dim=32,
                embed_matrix=pretrained_embeddings,
                padding_idx=0,
            )
        elif model_name == "stacked":
            model = StackedAttentiveRNN(
                vocab_size=vocab_size,
                embed_dim=32,
                embed_matrix=pretrained_embeddings,
                padding_idx=0,
                n_blocks=2,
            )
    out = model(torch.from_numpy(padded_sequences))
    assert out.size(0) == 100 and out.size(1) == 64


###############################################################################
# Without Pretrained Embeddings and head layers
###############################################################################


@pytest.mark.parametrize(
    "attention",
    [True, False],
)
def test_model_with_head_layers(attention):
    if not attention:
        model = BasicRNN(
            vocab_size=vocab_size,
            embed_dim=32,
            padding_idx=0,
            head_hidden_dims=[64, 16],
        )
    else:
        model = AttentiveRNN(
            vocab_size=vocab_size,
            embed_dim=32,
            padding_idx=0,
            head_hidden_dims=[64, 16],
        )
    out = model(torch.from_numpy(padded_sequences))
    assert out.size(0) == 100 and out.size(1) == 16


###############################################################################
# Pretrained Embeddings made non-trainable
###############################################################################


def test_embed_non_trainable():
    model = BasicRNN(
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
    model = BasicRNN(
        vocab_size=vocab_size,
        rnn_type="gru",
        embed_dim=32,
        bidirectional=bidirectional,
        padding_idx=0,
        use_hidden_state=False,
    )
    out = model(torch.from_numpy(padded_sequences))  # noqa: F841
    assert out.size(0) == 100 and out.size(1) == model.output_dim


# ###############################################################################
# # Test StackedAttentiveRNN
# ###############################################################################


@pytest.mark.parametrize(
    "rnn_type",
    ["lstm", "gru"],
)
@pytest.mark.parametrize(
    "bidirectional",
    [True, False],
)
@pytest.mark.parametrize(
    "attn_concatenate",
    [True, False],
)
@pytest.mark.parametrize(
    "with_addnorm",
    [True, False],
)
@pytest.mark.parametrize(
    "with_head",
    [True, False],
)
def test_stacked_attentive_rnn(
    rnn_type, bidirectional, attn_concatenate, with_addnorm, with_head
):

    model = StackedAttentiveRNN(
        vocab_size=vocab_size,
        embed_dim=32,
        hidden_dim=32,
        n_blocks=2,
        padding_idx=0,
        rnn_type=rnn_type,
        bidirectional=bidirectional,
        attn_concatenate=attn_concatenate,
        with_addnorm=with_addnorm,
        head_hidden_dims=[50] if with_head else None,
    )
    out = model(torch.from_numpy(padded_sequences))

    res = []
    res.append(out.size(0) == 100)

    if with_head:
        res.append(out.size(1) == 50)
    else:
        if bidirectional and attn_concatenate:
            out_dim = model.hidden_dim * 4
        elif bidirectional or attn_concatenate:
            out_dim = model.hidden_dim * 2
        else:
            out_dim = model.hidden_dim
        res.append(out.size(1) == out_dim)

    assert all(res)


def test_stacked_attentive_rnn_embed_non_trainable():
    model = StackedAttentiveRNN(
        vocab_size=vocab_size,
        embed_matrix=pretrained_embeddings,
        embed_trainable=False,
        n_blocks=2,
        padding_idx=0,
    )
    out = model(torch.from_numpy(padded_sequences))  # noqa: F841
    assert np.allclose(model.word_embed.weight.numpy(), pretrained_embeddings)


# ###############################################################################
# # Test Attn weights are ok
# ###############################################################################


@pytest.mark.parametrize(
    "stacked",
    [True, False],
)
def test_attn_weights(stacked):
    if stacked:
        model = StackedAttentiveRNN(
            vocab_size=vocab_size,
            embed_dim=32,
            n_blocks=2,
            padding_idx=0,
        )
    else:
        model = AttentiveRNN(
            vocab_size=vocab_size,
            embed_dim=32,
            padding_idx=0,
            head_hidden_dims=[64, 16],
        )

    out = model(torch.from_numpy(padded_sequences))  # noqa: F841

    attn_w = model.attention_weights

    if stacked:
        assert len(attn_w) == model.n_blocks and attn_w[0].size() == torch.Size(
            [100, 50]
        )
    else:
        assert attn_w.size() == torch.Size([100, 50])
