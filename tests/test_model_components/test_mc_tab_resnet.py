import string

import numpy as np
import torch
import pytest

from pytorch_widedeep.models import TabResnet

colnames = list(string.ascii_lowercase)[:10]
embed_cols = [np.random.choice(np.arange(5), 10) for _ in range(5)]
embed_input = [(u, i, j) for u, i, j in zip(colnames[:5], [5] * 5, [16] * 5)]
cont_cols = [np.random.rand(10) for _ in range(5)]
continuous_cols = colnames[-5:]

X_tab = torch.from_numpy(np.vstack(embed_cols + cont_cols).transpose())
X_tab_emb = X_tab[:, :5]
X_tab_cont = X_tab[:, 5:]


###############################################################################
# Embeddings and no continuous_cols
###############################################################################


@pytest.mark.parametrize(
    "simplify_blocks",
    [
        True,
        False,
    ],
)
def test_tab_resnet_cat(simplify_blocks):
    model = TabResnet(
        column_idx={k: v for v, k in enumerate(colnames[:5])},
        cat_embed_input=embed_input,
        blocks_dims=[32, 16],
        blocks_dropout=0.5,
        simplify_blocks=simplify_blocks,
    )
    out = model(X_tab_emb)
    assert out.size(0) == 10 and out.size(1) == 16


###############################################################################
# Cont and no cat
###############################################################################


@pytest.mark.parametrize(
    "simplify_blocks",
    [
        True,
        False,
    ],
)
@pytest.mark.parametrize(
    "embed_continuous",
    [
        True,
        False,
    ],
)
def test_tab_resnet_cont(simplify_blocks, embed_continuous):
    model = TabResnet(
        column_idx={k: v for v, k in enumerate(colnames[5:])},
        continuous_cols=continuous_cols,
        embed_continuous=embed_continuous,
        blocks_dims=[32, 16, 8],
        blocks_dropout=0.5,
    )
    out = model(X_tab)
    assert out.size(0) == 10 and out.size(1) == 8


###############################################################################
# Cat, Cont and an MLP
###############################################################################


@pytest.mark.parametrize(
    "use_bias",
    [
        True,
        False,
    ],
)
@pytest.mark.parametrize(
    "simplify_blocks",
    [
        True,
        False,
    ],
)
@pytest.mark.parametrize(
    "embed_continuous",
    [
        True,
        False,
    ],
)
def test_full_model(use_bias, simplify_blocks, embed_continuous):
    model = TabResnet(
        column_idx={k: v for v, k in enumerate(colnames)},
        cat_embed_input=embed_input,
        use_cat_bias=use_bias,
        continuous_cols=continuous_cols,
        use_cont_bias=use_bias,
        embed_continuous=embed_continuous,
        blocks_dims=[64, 32],
        blocks_dropout=0.5,
        simplify_blocks=simplify_blocks,
        mlp_dropout=0.5,
        mlp_hidden_dims=[16, 8],
    )
    out = model(X_tab)

    assert out.size(0) == 10 and out.size(1) == model.output_dim
