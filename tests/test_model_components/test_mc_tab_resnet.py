import string

import numpy as np
import torch
import pytest

from pytorch_widedeep.models import TabResnet

colnames = list(string.ascii_lowercase)[:10]
embed_cols = [np.random.choice(np.arange(5), 10) for _ in range(5)]
cont_cols = [np.random.rand(10) for _ in range(5)]

X_tab = torch.from_numpy(np.vstack(embed_cols + cont_cols).transpose())
X_tab_emb = X_tab[:, :5]
X_tab_cont = X_tab[:, 5:]


###############################################################################
# Embeddings and no continuous_cols
###############################################################################
embed_input = [(u, i, j) for u, i, j in zip(colnames[:5], [5] * 5, [16] * 5)]
model1 = TabResnet(
    blocks_dims=[32, 16],
    blocks_dropout=0.5,
    mlp_dropout=0.5,
    column_idx={k: v for v, k in enumerate(colnames[:5])},
    embed_input=embed_input,
)


def test_tab_resnet_embed():
    out = model1(X_tab_emb)
    assert out.size(0) == 10 and out.size(1) == 16


###############################################################################
# Continous Cols and Embeddings
###############################################################################
continuous_cols = colnames[-5:]
model2 = TabResnet(
    blocks_dims=[32, 16, 8],
    blocks_dropout=0.5,
    mlp_dropout=0.5,
    column_idx={k: v for v, k in enumerate(colnames)},
    embed_input=embed_input,
    continuous_cols=continuous_cols,
)


def test_tab_resnet_dense():
    out = model2(X_tab)
    assert out.size(0) == 10 and out.size(1) == 8


###############################################################################
# Continous Cols concatenated with Embeddings or with the output of the
# dense_resnet
###############################################################################
continuous_cols = colnames[-5:]


@pytest.mark.parametrize(
    "concat_cont_first",
    [
        True,
        False,
    ],
)
def test_cont_contat(concat_cont_first):
    model3 = TabResnet(
        blocks_dims=[32, 16, 8],
        blocks_dropout=0.5,
        mlp_dropout=0.5,
        column_idx={k: v for v, k in enumerate(colnames)},
        embed_input=embed_input,
        continuous_cols=continuous_cols,
        concat_cont_first=concat_cont_first,
    )
    out = model3(X_tab)

    assert out.size(0) == 10 and out.size(1) == model3.output_dim


###############################################################################
# Test full set up
###############################################################################


@pytest.mark.parametrize(
    "concat_cont_first, cont_norm_layer",
    [
        [True, None],
        [False, None],
        [True, "batchnorm"],
        [False, "batchnorm"],
        [True, "layernorm"],
        [False, "layernorm"],
    ],
)
def test_full_setup(concat_cont_first, cont_norm_layer):
    model4 = TabResnet(
        embed_input=embed_input,
        column_idx={k: v for v, k in enumerate(colnames)},
        blocks_dims=[32, 16, 8],
        blocks_dropout=0.5,
        mlp_dropout=0.5,
        mlp_hidden_dims=[32, 16],
        mlp_batchnorm=True,
        mlp_batchnorm_last=False,
        embed_dropout=0.1,
        continuous_cols=continuous_cols,
        cont_norm_layer=cont_norm_layer,
        concat_cont_first=concat_cont_first,
    )
    out = model4(X_tab)

    true_mlp_inp_dim = list(model4.tab_resnet_mlp.mlp.dense_layer_0.parameters())[
        2
    ].size(1)

    if concat_cont_first:
        expected_mlp_inp_dim = model4.blocks_dims[-1]
    else:
        expected_mlp_inp_dim = model4.blocks_dims[-1] + len(continuous_cols)

    assert (
        out.size(0) == 10
        and out.size(1) == model4.output_dim
        and expected_mlp_inp_dim == true_mlp_inp_dim
    )
