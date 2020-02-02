import torch

from pytorch_widedeep.models import Wide

inp = torch.rand(10, 10)
model = Wide(10, 1)


###############################################################################
# Simply testing that it runs
###############################################################################
def test_wide():
    out = model(inp)
    assert out.size(0) == 10 and out.size(1) == 1
