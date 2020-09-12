from copy import deepcopy

import pytest
from torch import nn

from pytorch_widedeep.models import (
    Wide,
    DeepText,
    WideDeep,
    DeepDense,
    DeepImage,
)

embed_input = [(u, i, j) for u, i, j in zip(["a", "b", "c"][:4], [4] * 3, [8] * 3)]
deep_column_idx = {k: v for v, k in enumerate(["a", "b", "c"])}
wide = Wide(10, 1)
deepdense = DeepDense(
    hidden_layers=[16, 8], deep_column_idx=deep_column_idx, embed_input=embed_input
)
deeptext = DeepText(vocab_size=100, embed_dim=8)
deepimage = DeepImage(pretrained=False)

###############################################################################
#  test raising 'output dim errors'
###############################################################################


@pytest.mark.parametrize(
    "deepcomponent, component_name",
    [
        (None, "dense"),
        (deeptext, "text"),
        (deepimage, "image"),
    ],
)
def test_history_callback(deepcomponent, component_name):
    if deepcomponent is None:
        deepcomponent = deepcopy(deepdense)
    deepcomponent.__dict__.pop("output_dim")
    with pytest.raises(AttributeError):
        if component_name == "dense":
            model = WideDeep(wide, deepdense=deepcomponent)
        elif component_name == "text":
            model = WideDeep(wide, deepdense=deepdense, deeptext=deepcomponent)
        elif component_name == "image":
            model = WideDeep(  # noqa: F841
                wide, deepdense=deepdense, deepimage=deepcomponent
            )


###############################################################################
#  test warning on head_layers and deephead
###############################################################################


def test_deephead_and_head_layers():
    deephead = nn.Sequential(nn.Linear(32, 16), nn.Linear(16, 8))
    with pytest.warns(UserWarning):
        model = WideDeep(  # noqa: F841
            wide=wide, deepdense=deepdense, head_layers=[16, 8], deephead=deephead
        )


###############################################################################
#  test deephead is None and head_layers is not None
###############################################################################


def test_no_deephead_and_head_layers():
    out = []
    model = WideDeep(wide=wide, deepdense=deepdense, head_layers=[8, 4])  # noqa: F841
    for n, p in model.named_parameters():
        if n == "deephead.head_layer_0.0.weight":
            out.append(p.size(0) == 8 and p.size(1) == 8)
        if n == "deephead.head_layer_1.0.weight":
            out.append(p.size(0) == 4 and p.size(1) == 8)
    assert all(out)
