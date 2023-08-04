import copy
import timeit

import torch
import pytest

from pytorch_widedeep.models.tabular.transformers._attention_layers import (
    MultiHeadedAttention,
)

torch.backends.cudnn.deterministic = True

input_dim = 128
n_heads = 4
device = "cuda" if torch.cuda.is_available() else "cpu"

standard_attn = (
    MultiHeadedAttention(
        input_dim=input_dim, n_heads=n_heads, use_bias=False, dropout=0
    )
    .to(device)
    .eval()
)

flash_attn = (
    MultiHeadedAttention(
        input_dim=input_dim,
        n_heads=n_heads,
        use_bias=False,
        dropout=0,
        use_flash_attention=True,
    )
    .to(device)
    .eval()
)

linear_attn = (
    MultiHeadedAttention(
        input_dim=input_dim,
        n_heads=n_heads,
        use_bias=False,
        dropout=0,
        use_linear_attention=True,
    )
    .to(device)
    .eval()
)


# Set initialization weights equal for comparison
for module in ["q_proj", "kv_proj", "out_proj"]:
    getattr(flash_attn, module).weight = copy.deepcopy(
        getattr(standard_attn, module).weight
    )

X = torch.randn(128, 100, input_dim).to(device)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA to run")
def test_flash_standard_shapes():
    # Check that shapes of output are the same
    assert standard_attn(X).shape == flash_attn(X).shape


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA to run")
def test_flash_standard_values():
    # Check output values match between both implementations
    assert torch.allclose(standard_attn(X), flash_attn(X), atol=1e-7)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA to run")
def test_speedup_flash():
    # Check that iterations happen faster
    standard_its = timeit.timeit(lambda: standard_attn(X), number=500)
    flash_its = timeit.timeit(lambda: flash_attn(X), number=500)

    assert standard_its > flash_its
    assert ((standard_its - flash_its) / standard_its) > 0.3


def test_flash_standard_vs_linear_shapes():
    # Check that shapes of output are the same
    assert standard_attn(X).shape == linear_attn(X).shape
