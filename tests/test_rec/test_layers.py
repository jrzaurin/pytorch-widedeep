import torch
import pytest

from pytorch_widedeep.models.rec.din import Dice, ActivationUnit
from pytorch_widedeep.models.rec.xdeepfm import CompressedInteractionNetwork


@pytest.fixture
def batch_size():
    return 32


@pytest.fixture
def embed_dim():
    return 16


@pytest.fixture
def seq_len():
    return 10


@pytest.fixture
def num_cols():
    return 5


def test_dice(batch_size, embed_dim, seq_len):
    dice = Dice(input_dim=embed_dim)

    # random tensor with non negative values
    X = torch.randn(batch_size, seq_len, embed_dim)
    X = X.clip(min=0)

    output = dice(X)

    assert output.shape == (batch_size, seq_len, embed_dim)
    assert torch.all(output >= 0) and torch.all(output <= X)


def test_dice_backward(batch_size, embed_dim, seq_len):
    dice = Dice(input_dim=embed_dim)
    X = torch.randn(batch_size, seq_len, embed_dim, requires_grad=True)

    output = dice(X)
    loss = output.sum()
    loss.backward()

    assert X.grad is not None
    assert X.grad.shape == X.shape


@pytest.mark.parametrize("activation", ["prelu", "dice"])
def test_activation_unit(batch_size, embed_dim, seq_len, activation):
    au = ActivationUnit(embed_dim=embed_dim, activation=activation)
    item = torch.randn(batch_size, 1, embed_dim)
    user_behavior = torch.randn(batch_size, seq_len, embed_dim)

    output = au(item, user_behavior)

    assert output.shape == (batch_size, seq_len)
    assert torch.allclose(output.sum(dim=1), torch.ones(batch_size))


def test_activation_unit_backward(batch_size, embed_dim, seq_len):
    au = ActivationUnit(embed_dim=embed_dim, activation="prelu")
    item = torch.randn(batch_size, 1, embed_dim, requires_grad=True)
    user_behavior = torch.randn(batch_size, seq_len, embed_dim, requires_grad=True)

    output = au(item, user_behavior)
    loss = output.sum()
    loss.backward()

    assert item.grad is not None
    assert user_behavior.grad is not None
    assert item.grad.shape == item.shape
    assert user_behavior.grad.shape == user_behavior.shape


def test_cin(batch_size, embed_dim, num_cols):
    cin_layer_dims = [16, 8]
    cin = CompressedInteractionNetwork(num_cols=num_cols, cin_layer_dims=cin_layer_dims)
    X = torch.randn(batch_size, num_cols, embed_dim)

    output = cin(X)

    expected_output_dim = sum(cin_layer_dims)
    assert output.shape == (batch_size, expected_output_dim)


def test_cin_backward(batch_size, embed_dim, num_cols):
    cin_layer_dims = [64, 32]
    cin = CompressedInteractionNetwork(num_cols=num_cols, cin_layer_dims=cin_layer_dims)
    X = torch.randn(batch_size, num_cols, embed_dim, requires_grad=True)

    output = cin(X)
    loss = output.sum()
    loss.backward()

    assert X.grad is not None
    assert X.grad.shape == X.shape
