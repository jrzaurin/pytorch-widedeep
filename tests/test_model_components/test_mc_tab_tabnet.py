import string

import numpy as np
import torch
import pytest

from pytorch_widedeep.wdtypes import WideDeep
from pytorch_widedeep.models.tabnet._utils import create_explain_matrix
from pytorch_widedeep.models.tabnet.tab_net import TabNet  # noqa: F403

# I am going over test this model due to the number of components

n_embed = 5
# this is the number of embed_cols and cont_cols. So total num of cols =
# n_cols * 2
n_cols = 2
batch_size = 10
colnames = list(string.ascii_lowercase)[: (n_cols * 2)]
embed_cols = [np.random.choice(np.arange(n_embed), batch_size) for _ in range(n_cols)]
cont_cols = [np.random.rand(batch_size) for _ in range(n_cols)]

X_tab = torch.from_numpy(np.vstack(embed_cols + cont_cols).transpose())
X_tab_emb = X_tab[:, :n_cols]
X_tab_cont = X_tab[:, n_cols:]

###############################################################################
# Test functioning using the defaults
###############################################################################

embed_input = [(u, i, 1) for u, i in zip(colnames[:2], [n_embed] * 2)]

model1 = TabNet(
    column_idx={k: v for v, k in enumerate(colnames)},
    embed_input=embed_input,
    continuous_cols=colnames[n_cols:],
)


def test_embeddings_have_padding():
    res = []
    for k, v in model1.cat_embed_and_cont.embed_layers.items():
        res.append(v.weight.size(0) == n_embed + 1)
        res.append(not torch.all(v.weight[0].bool()))
    assert all(res)


@pytest.mark.parametrize(
    "cont_norm_layer",
    [
        None,
        "batchnorm",
        "layernorm",
    ],
)
def test_tabnet_output(cont_norm_layer):
    model = TabNet(
        column_idx={k: v for v, k in enumerate(colnames)},
        embed_input=embed_input,
        continuous_cols=colnames[n_cols:],
        cont_norm_layer=cont_norm_layer,
    )
    out1, out2 = model(X_tab)
    assert out1.size(0) == 10 and out1.size(1) == model1.step_dim


###############################################################################
# Test functioning with different types of masks
###############################################################################


@pytest.mark.parametrize(
    "mask_type",
    [
        "sparsemax",
        "entmax",
    ],
)
def test_mask_type(mask_type):
    model = TabNet(
        column_idx={k: v for v, k in enumerate(colnames)},
        embed_input=embed_input,
        continuous_cols=colnames[n_cols:],
        mask_type=mask_type,
    )
    out1, out2 = model(X_tab)
    assert out1.size(0) == 10 and out1.size(1) == model1.step_dim


###############################################################################
# Test functioning with/without ghost BN
###############################################################################


@pytest.mark.parametrize(
    "ghost_bn",
    [
        True,
        False,
    ],
)
def test_ghost_bn(ghost_bn):
    model = TabNet(
        column_idx={k: v for v, k in enumerate(colnames)},
        embed_input=embed_input,
        continuous_cols=colnames[n_cols:],
        ghost_bn=ghost_bn,
    )
    out1, out2 = model(X_tab)
    assert out1.size(0) == 10 and out1.size(1) == model1.step_dim


###############################################################################
# Test forward_mask method
###############################################################################


def test_forward_masks():
    out1, out2 = model1.forward_masks(X_tab)
    bsz, nfeat = X_tab.shape[0], X_tab.shape[1]
    out = []
    out.append(out1.shape[0] == bsz)
    out.append(out1.shape[1] == nfeat)
    for step in range(model1.n_steps):
        out.append(out2[step].size(0) == bsz)
        out.append(out2[step].size(1) == nfeat)
    assert all(out)


###############################################################################
# Test create_explain_matrix
###############################################################################


def test_create_explain_matrix():
    embed_input = [(u, i, 2) for u, i in zip(colnames[:2], [n_embed] * 2)]
    continuous_cols = colnames[2:]
    embed_cols = colnames[:2]
    column_idx = {k: v for v, k in enumerate(colnames)}

    tabnet = TabNet(
        column_idx=column_idx,
        embed_input=embed_input,
        continuous_cols=continuous_cols,
    )
    wdmodel = WideDeep(deeptabular=tabnet)

    expl_mtx = create_explain_matrix(wdmodel)

    checks = []
    checks.append(expl_mtx.sum() == tabnet.embed_and_cont_dim)
    checks.append(all(expl_mtx.sum(1) == 1))
    for col, idx in column_idx.items():
        if col in embed_cols:
            checks.append(expl_mtx[:, idx].sum() == 2.0)
        elif col in continuous_cols:
            checks.append(expl_mtx[:, idx].sum() == 1.0)

    assert all(checks)
