import copy
import torch
import timeit
import pytest

import torch.backends.cuda as tcud
from pytorch_widedeep.models.tabular.transformers._attention_layers import MultiHeadedAttention, _flash_kernel_setup, SDPBackend


torch.backends.cudnn.deterministic = True

input_dim = 128
n_heads = 4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'

standard_net = MultiHeadedAttention(
    input_dim=input_dim,
    n_heads=n_heads,
    use_bias=False,
    dropout=0
).to(device).eval()

flash_net = MultiHeadedAttention(
    input_dim=input_dim,
    n_heads=n_heads,
    use_bias=False,
    dropout=0,
    use_flash=True
).to(device).eval()

# Set initialization weights equal for comparison
for module in ['q_proj', 'kv_proj', 'out_proj']:
    getattr(flash_net, module).weight = copy.deepcopy(getattr(standard_net, module).weight)

X = torch.randn(128, 100, input_dim).to(device)

@pytest.mark.parametrize(
    "backends, active", [
        ([SDPBackend.FLASH], {
            tcud.flash_sdp_enabled: True, 
            tcud.mem_efficient_sdp_enabled: False,
            tcud.math_sdp_enabled: False
        }),
        ([SDPBackend.MEM_EFFICIENT], {
            tcud.flash_sdp_enabled: False, 
            tcud.mem_efficient_sdp_enabled: True,
            tcud.math_sdp_enabled: False
        }),
        ([SDPBackend.FLASH, SDPBackend.MEM_EFFICIENT], {
            tcud.flash_sdp_enabled: True, 
            tcud.mem_efficient_sdp_enabled: True,
            tcud.math_sdp_enabled: False
        })
    ]
)
def test_cdp_context_managment(backends, active):
    ctx = _flash_kernel_setup(backends)
    with ctx:
        assert all([f() == v for f, v in active.items()])
        
def test_flash_standard_shapes():
    # Check that shapes of output are the same
    assert standard_net(X).shape == flash_net(X).shape

def test_flash_standard_values():
    # Check output values match between both implementations
    assert torch.allclose(standard_net(X), flash_net(X), atol=1e-7)

def test_speedup_flash():
    # Check that iterations happen faster
    standard_its = timeit.timeit(lambda: standard_net(X), number=500)
    flash_its = timeit.timeit(lambda: flash_net(X), number=500)

    assert standard_its > flash_its
    assert ((standard_its - flash_its) / standard_its) > 0.3
