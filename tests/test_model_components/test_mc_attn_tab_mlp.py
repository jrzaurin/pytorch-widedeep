import string

import numpy as np
import torch
import pandas as pd
import pytest

from pytorch_widedeep.models import AttentiveTabMlp
from pytorch_widedeep.preprocessing import TabPreprocessor

colnames = list(string.ascii_lowercase)[:10]
embed_cols = [np.random.choice(np.arange(5), 10) for _ in range(5)]
embed_input = [(u, i, j) for u, i, j in zip(colnames[:5], [5] * 5, [16] * 5)]
cont_cols = [np.random.rand(10) for _ in range(5)]
continuous_cols = colnames[-5:]

X_deep = torch.from_numpy(np.vstack(embed_cols + cont_cols).transpose())
X_deep_emb = X_deep[:, :5]
X_deep_cont = X_deep[:, 5:]

X_deep_with_class = torch.from_numpy(np.hstack([np.ones((10, 1)) * 6, X_deep]))

###############################################################################
# Embeddings and NO continuous_cols
###############################################################################


@pytest.mark.parametrize(
    "attention_name",
    ["context_attention", "self_attention"],
)
@pytest.mark.parametrize(
    "with_mlp",
    [True, False],
)
def test_only_cat_embed(attention_name, with_mlp):
    if with_mlp:
        mlp_hidden_dims = [32, 16]
    else:
        mlp_hidden_dims = None

    model = AttentiveTabMlp(
        column_idx={k: v for v, k in enumerate(colnames[:5])},
        cat_embed_input=embed_input,
        attention_name=attention_name,
        mlp_hidden_dims=mlp_hidden_dims,
    )

    out = model(X_deep_emb)

    assert out.size(0) == 10 and out.size(1) == model.output_dim


# ###############################################################################
# # Continous cols but NO embeddings
# ###############################################################################


@pytest.mark.parametrize(
    "attention_name",
    ["context_attention", "self_attention"],
)
@pytest.mark.parametrize(
    "with_mlp",
    [True, False],
)
def test_only_cont_embed(attention_name, with_mlp):
    if with_mlp:
        mlp_hidden_dims = [32, 16]
    else:
        mlp_hidden_dims = None

    model = AttentiveTabMlp(
        column_idx={k: v for v, k in enumerate(colnames[5:])},
        continuous_cols=continuous_cols,
        attention_name=attention_name,
        mlp_hidden_dims=mlp_hidden_dims,
    )

    out = model(X_deep_cont)

    assert out.size(0) == 10 and out.size(1) == model.output_dim


# ###############################################################################
# # All parameters and cont norm
# ###############################################################################


@pytest.mark.parametrize(
    "cont_norm_layer",
    [None, "batchnorm", "layernorm"],
)
@pytest.mark.parametrize(
    "attention_name",
    ["context_attention", "self_attention"],
)
@pytest.mark.parametrize(
    "with_mlp",
    [True, False],
)
def test_cat_and_cont_embed(cont_norm_layer, attention_name, with_mlp):
    if with_mlp:
        mlp_hidden_dims = [32, 16]
    else:
        mlp_hidden_dims = None

    model = AttentiveTabMlp(
        column_idx={k: v for v, k in enumerate(colnames)},
        cat_embed_input=embed_input,
        continuous_cols=continuous_cols,
        cont_norm_layer=cont_norm_layer,
        attention_name=attention_name,
        mlp_hidden_dims=mlp_hidden_dims,
    )

    out = model(X_deep)

    assert out.size(0) == 10 and out.size(1) == model.output_dim


# ###############################################################################
# # Test shared embeddings and with cls token and attention weights
# ###############################################################################


df = pd.DataFrame(
    {
        "col1": ["a", "b", "c"],
        "col2": ["c", "d", "e"],
        "col3": [10, 20, 30],
        "col4": [2, 7, 9],
    }
)


@pytest.mark.parametrize(
    "attention_name",
    ["context_attention", "self_attention"],
)
@pytest.mark.parametrize(
    "shared_embed",
    [True, False],
)
@pytest.mark.parametrize(
    "with_cls_token",
    [True, False],
)
def test_shared_embed_and_cls(attention_name, shared_embed, with_cls_token):

    tab_preprocessor = TabPreprocessor(
        embed_cols=["col1", "col2"],
        continuous_cols=["col3", "col4"],
        with_attention=True,
        with_cls_token=with_cls_token,
        verbose=False,
    )

    X = tab_preprocessor.fit_transform(df)
    X_inp = torch.from_numpy(X)

    model = AttentiveTabMlp(
        column_idx=tab_preprocessor.column_idx,
        cat_embed_input=tab_preprocessor.embeddings_input,
        continuous_cols=["col3", "col4"],
        attention_name=attention_name,
    )

    out = model(X_inp)
    attn_weights = model.attention_weights

    checks = []
    checks.append(len(attn_weights) == model.n_blocks)
    checks.append(out.size(1) == model.output_dim)

    if attention_name == "context_attention":
        s0 = df.shape[0]
        s1 = df.shape[1] + 1 if with_cls_token else df.shape[1]
        checks.append(attn_weights[0].size() == torch.Size((s0, s1)))
    elif attention_name == "self_attention":
        s0 = df.shape[0]
        s1 = df.shape[1] + 1 if with_cls_token else df.shape[1]
        checks.append(attn_weights[0].size() == torch.Size((s0, model.n_heads, s1, s1)))

    assert all(checks)
