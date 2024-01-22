import string
from random import choices

import numpy as np
import pandas as pd
import pytest

from pytorch_widedeep import Tab2Vec
from pytorch_widedeep.models import (
    SAINT,
    TabMlp,
    TabNet,
    WideDeep,
    TabResnet,
    TabPerceiver,
    FTTransformer,
    TabFastFormer,
    TabTransformer,
    SelfAttentionMLP,
    ContextAttentionMLP,
)
from pytorch_widedeep.preprocessing import TabPreprocessor

colnames = list(string.ascii_lowercase)[:4] + ["target"]
cat_col1_vals = ["a", "b", "c"]
cat_col2_vals = ["d", "e", "f"]


def create_df():
    cat_cols = [np.array(choices(c, k=5)) for c in [cat_col1_vals, cat_col2_vals]]
    cont_cols = [np.round(np.random.rand(5), 2) for _ in range(2)]
    target = [np.random.choice(2, 5, p=[0.8, 0.2])]
    return pd.DataFrame(
        np.vstack(cat_cols + cont_cols + target).transpose(), columns=colnames
    )


df_init = create_df()
df_t2v = create_df()

embed_cols = [("a", 2), ("b", 4)]
cont_cols = ["c", "d"]
tab_preprocessor = TabPreprocessor(cat_embed_cols=embed_cols, continuous_cols=cont_cols)
X_tab = tab_preprocessor.fit_transform(df_init)

tabmlp = TabMlp(
    column_idx=tab_preprocessor.column_idx,
    cat_embed_input=tab_preprocessor.cat_embed_input,
    continuous_cols=tab_preprocessor.continuous_cols,
    mlp_hidden_dims=[8, 4],
)

tabresnet = TabResnet(
    column_idx=tab_preprocessor.column_idx,
    cat_embed_input=tab_preprocessor.cat_embed_input,
    continuous_cols=tab_preprocessor.continuous_cols,
    blocks_dims=[8, 8, 4],
)

tabnet = TabNet(
    column_idx=tab_preprocessor.column_idx,
    cat_embed_input=tab_preprocessor.cat_embed_input,
    continuous_cols=tab_preprocessor.continuous_cols,
)


@pytest.mark.parametrize(
    "deeptabular, return_dataframe",
    [
        (tabmlp, True),
        (tabmlp, False),
        (tabresnet, True),
        (tabresnet, False),
        (tabnet, True),
        (tabnet, False),
    ],
)
def test_non_transformer_models(deeptabular, return_dataframe):
    model = WideDeep(deeptabular=deeptabular)

    # Let's assume the model is trained
    t2v = Tab2Vec(
        tab_preprocessor=tab_preprocessor,
        model=model,
        return_dataframe=return_dataframe,
    )
    t2v_out, _ = t2v.transform(df_t2v, target_col="target")

    embed_dim = sum([el[2] for el in tab_preprocessor.cat_embed_input])
    cont_dim = len(tab_preprocessor.continuous_cols)
    assert t2v_out.shape[1] == embed_dim + cont_dim


###############################################################################
# Test Transformer models
###############################################################################


def _build_model(model_name, params):
    if model_name == "tabtransformer":
        return TabTransformer(input_dim=8, n_heads=2, n_blocks=2, **params)
    if model_name == "saint":
        return SAINT(input_dim=8, n_heads=2, n_blocks=2, **params)
    if model_name == "fttransformer":
        return FTTransformer(n_blocks=2, n_heads=2, kv_compression_factor=0.5, **params)
    if model_name == "tabfastformer":
        return TabFastFormer(n_blocks=2, n_heads=2, **params)
    if model_name == "tabperceiver":
        return TabPerceiver(
            input_dim=8,
            n_cross_attn_heads=2,
            n_latents=2,
            latent_dim=8,
            n_latent_heads=2,
            n_perceiver_blocks=2,
            share_weights=False,
            **params
        )


@pytest.mark.parametrize(
    "model_name, with_cls_token, share_embeddings, embed_continuous",
    [
        ("tabtransformer", False, False, False),
        ("tabtransformer", True, False, False),
        ("tabtransformer", False, True, False),
        ("tabtransformer", True, False, True),
    ],
)
def test_tab_transformer_models(
    model_name, with_cls_token, share_embeddings, embed_continuous
):
    embed_cols = ["a", "b"]
    cont_cols = ["c", "d"]

    tab_preprocessor = TabPreprocessor(
        cat_embed_cols=embed_cols,
        continuous_cols=cont_cols,
        for_transformer=True,
        with_cls_token=with_cls_token,
        shared_embed=share_embeddings,
    )
    X_tab = tab_preprocessor.fit_transform(df_init)  # noqa: F841

    params = {
        "column_idx": tab_preprocessor.column_idx,
        "cat_embed_input": tab_preprocessor.cat_embed_input,
        "continuous_cols": tab_preprocessor.continuous_cols,
        "embed_continuous": embed_continuous,
        "embed_continuous_method": "standard" if embed_continuous else None,
    }

    deeptabular = _build_model(model_name, params)

    # Let's assume the model is trained
    model = WideDeep(deeptabular=deeptabular)
    t2v = Tab2Vec(
        tab_preprocessor=tab_preprocessor,
        model=model,
    )
    x_vec = t2v.transform(df_t2v)

    if embed_continuous:
        out_dim = (len(embed_cols) + len(cont_cols)) * deeptabular.input_dim
    else:
        out_dim = len(embed_cols) * deeptabular.input_dim + len(cont_cols)

    assert x_vec.shape[1] == out_dim


###############################################################################
# Test SelfAttentionMLP and ContextAttentionMLP
###############################################################################


@pytest.mark.parametrize(
    "with_cls_token",
    [True, False],
)
@pytest.mark.parametrize(
    "share_embeddings",
    [True, False],
)
@pytest.mark.parametrize(
    "attention_name",
    ["context_attention", "self_attention"],
)
def test_attentive_mlp(
    with_cls_token,
    share_embeddings,
    attention_name,
):
    embed_cols = ["a", "b"]
    cont_cols = ["c", "d"]

    tab_preprocessor = TabPreprocessor(
        cat_embed_cols=embed_cols,
        continuous_cols=cont_cols,
        with_attention=True,
        with_cls_token=with_cls_token,
        shared_embed=share_embeddings,
    )
    X_tab = tab_preprocessor.fit_transform(df_init)  # noqa: F841

    if attention_name == "context_attention":
        deeptabular = ContextAttentionMLP(
            column_idx=tab_preprocessor.column_idx,
            cat_embed_input=tab_preprocessor.cat_embed_input,
            continuous_cols=tab_preprocessor.continuous_cols,
        )

    elif attention_name == "self_attention":
        deeptabular = SelfAttentionMLP(
            column_idx=tab_preprocessor.column_idx,
            cat_embed_input=tab_preprocessor.cat_embed_input,
            continuous_cols=tab_preprocessor.continuous_cols,
        )

    # Let's assume the model is trained
    model = WideDeep(deeptabular=deeptabular)
    t2v = Tab2Vec(
        tab_preprocessor=tab_preprocessor,
        model=model,
    )
    x_vec = t2v.transform(df_t2v)

    out_dim = (len(embed_cols) + len(cont_cols)) * deeptabular.input_dim

    assert x_vec.shape[1] == out_dim


@pytest.mark.parametrize(
    "model_name, with_cls_token, share_embeddings, return_dataframe",
    [
        ("saint", False, True, False),
        ("saint", True, True, False),
        ("saint", False, False, False),
        ("saint", False, True, True),
        ("saint", True, True, True),
        ("saint", False, False, True),
        ("fttransformer", False, True, False),
        ("fttransformer", True, True, False),
        ("fttransformer", False, False, False),
        ("fttransformer", False, True, True),
        ("fttransformer", True, True, True),
        ("fttransformer", False, False, True),
        ("tabfastformer", False, True, False),
        ("tabfastformer", True, True, False),
        ("tabfastformer", False, False, False),
        ("tabfastformer", False, True, True),
        ("tabfastformer", True, True, True),
        ("tabfastformer", False, False, True),
        (
            "tabperceiver",
            False,
            True,
            False,
        ),  # for the perceiver we do not need with_cls_token
        ("tabperceiver", False, False, False),
        (
            "tabperceiver",
            False,
            True,
            False,
        ),
        ("tabperceiver", False, False, True),
    ],
)
def test_transformer_family_models(
    model_name, with_cls_token, share_embeddings, return_dataframe
):
    embed_cols = ["a", "b"]
    cont_cols = ["c", "d"]

    tab_preprocessor = TabPreprocessor(
        cat_embed_cols=embed_cols,
        continuous_cols=cont_cols,
        for_transformer=True,
        with_cls_token=with_cls_token,
        shared_embed=share_embeddings,
    )
    X_tab = tab_preprocessor.fit_transform(df_init)  # noqa: F841

    params = {
        "column_idx": tab_preprocessor.column_idx,
        "cat_embed_input": tab_preprocessor.cat_embed_input,
        "continuous_cols": tab_preprocessor.continuous_cols,
    }

    deeptabular = _build_model(model_name, params)

    # Let's assume the model is trained
    model = WideDeep(deeptabular=deeptabular)
    t2v = Tab2Vec(
        tab_preprocessor=tab_preprocessor,
        model=model,
        return_dataframe=return_dataframe,
    )
    x_vec = t2v.transform(df_t2v)

    out_dim = (len(embed_cols) + len(cont_cols)) * deeptabular.input_dim

    assert x_vec.shape[1] == out_dim
