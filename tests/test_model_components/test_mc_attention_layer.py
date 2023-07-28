import torch
import timeit

from pytorch_widedeep.models.tabular.transformers._attention_layers import MultiHeadedAttention

input_dim = 128
n_heads = 4

m = MultiHeadedAttention(
    input_dim=input_dim,
    n_heads=n_heads,
    use_bias=False,
    dropout=0.2
)

mf = MultiHeadedAttention(
    input_dim=input_dim,
    n_heads=n_heads,
    use_bias=False,
    dropout=0.2,
    use_flash=True
)

X = torch.randn(32, 10, input_dim)

def test_flash_standard_shapes():
    # Check that shapes of output are the same
    assert m(X).shape == mf(X).shape

def test_speedup_flash():
    # Check that iterations happen faster
    standard_its = timeit.timeit(lambda: m(X), number=3000)
    flash_its = timeit.timeit(lambda: mf(X), number=3000)

    assert standard_its > flash_its
    assert ((standard_its - flash_its) / standard_its) > 0.1
