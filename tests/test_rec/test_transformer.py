# ###############################################################################
# # Test Basic Transformer
# ###############################################################################

import numpy as np
import torch
import pytest

from pytorch_widedeep.models.rec import Transformer

padded_sequences = np.random.choice(np.arange(1, 100), (100, 48))
padded_sequences = np.hstack(
    (np.repeat(np.array([[0, 0]]), 100, axis=0), padded_sequences)
)


@pytest.mark.parametrize(
    "with_cls_token",
    [True, False],
)
def test_basic_transformer(with_cls_token):
    if with_cls_token:
        # if we use a 'CLS' token it must be inserted at the beginning of the
        # sequence
        _padded_sequences = np.zeros(
            (padded_sequences.shape[0], padded_sequences.shape[1] + 1), dtype=int
        )
        _padded_sequences[:, 0] = padded_sequences.max() + 1
        _padded_sequences[:, 1:] = padded_sequences
    else:
        _padded_sequences = padded_sequences

    model = Transformer(
        vocab_size=_padded_sequences.max() + 1,
        seq_length=_padded_sequences.shape[1],
        input_dim=8,
        n_heads=2,
        n_blocks=2,
        with_pos_encoding=False,
        with_cls_token=with_cls_token,
    )

    out = model(torch.from_numpy(_padded_sequences))

    res = []
    res.append(out.size(0) == _padded_sequences.shape[0])
    res.append(out.size(1) == model.output_dim)

    assert all(res)


# ###############################################################################
# # Test Custom Positional Encoder
# ###############################################################################


class DummyPositionalEncoding(torch.nn.Module):
    def __init__(self, input_dim: int, seq_length: int):
        super().__init__()

        pe = torch.ones(1, seq_length, input_dim)
        self.register_buffer("pe", pe)

    def forward(self, X):
        return X + self.pe


def test_custom_pos_encoder():
    model = Transformer(
        vocab_size=padded_sequences.max() + 1,
        seq_length=padded_sequences.shape[1],
        input_dim=8,
        n_heads=2,
        n_blocks=2,
        pos_encoder=DummyPositionalEncoding(
            input_dim=8, seq_length=padded_sequences.shape[1]
        ),
    )

    out = model(torch.from_numpy(padded_sequences))

    res = []
    res.append(out.size(0) == padded_sequences.shape[0])
    res.append(out.size(1) == model.output_dim)

    assert all(res)
