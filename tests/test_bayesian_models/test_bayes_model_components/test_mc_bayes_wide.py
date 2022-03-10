import torch

from pytorch_widedeep.bayesian_models import BayesianWide

inp = torch.rand(10, 10)
model = BayesianWide(10, 1)


###############################################################################
# Simply testing that it runs
###############################################################################
def test_wide():
    out = model(inp)
    assert out.size(0) == 10 and out.size(1) == 1
