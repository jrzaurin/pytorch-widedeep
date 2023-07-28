import pytest
import torch
import timeit

from pytorch_widedeep.models.tabular.transformers._attention_layers import MultiHeadedAttention, _flash_kernel_setup, SDPBackend


input_dim = 128
n_heads = 4
device = 'cuda' if torch.cuda.is_available() else 'cpu'


import torch.backends.cuda as tcud

X = torch.randn(128, 100, input_dim).to(device)

@pytest.mark.parametrize(
    "backends, active", [
        ([SDPBackend.FLASH], {
            tcud.flash_sdp_enabled: True, 
            tcud.mem_efficient_sdp_enabled: False,
            tcud.math_sdp_enabled: False
        }),
        # ([SDPBackend.MEM_EFFICIENT],
        # [SDPBackend.FLASH, SDPBackend.MEM_EFFICIENT]
    ]
)
def test_cdp_context_managment(backends, active):
    ctx = _flash_kernel_setup(backends)
    with ctx:
        assert all([f() == v for f, v in active.items()])
        
# @pytest.mark.parametrize(
#     "backends", [
#         [SDPBackend.FLASH],
#         [SDPBackend.MEM_EFFICIENT],
#         [SDPBackend.FLASH, SDPBackend.MEM_EFFICIENT]
#     ]
# )
# def test_flash_standard_shapes(backends):
#     m = MultiHeadedAttention(
#         input_dim=input_dim,
#         n_heads=n_heads,
#         use_bias=False,
#         dropout=0
#     ).to(device)

#     mf = MultiHeadedAttention(
#         input_dim=input_dim,
#         n_heads=n_heads,
#         use_bias=False,
#         dropout=0,
#         use_flash=True,
#         enabled_flash_backends=backends
#     ).to(device)

#     # Eval mode to disable dropout
#     m = m.eval()
#     mf = mf.eval()

#     # Check that shapes of output are the same
#     assert m(X).shape == mf(X).shape

# def test_flash_standard_values():
#     # Check output values match between both implementations
#     assert torch.allclose(m(X), mf(X))

# # def test_flash_context_manager():


# def test_speedup_flash():
#     # Check that iterations happen faster
#     standard_its = timeit.timeit(lambda: m(X), number=3000)
#     flash_its = timeit.timeit(lambda: mf(X), number=3000)

#     print(standard_its, flash_its)
#     print((standard_its - flash_its) / standard_its)
#     assert 1 == 2
#     assert standard_its > flash_its
#     assert ((standard_its - flash_its) / standard_its) > 0.1
