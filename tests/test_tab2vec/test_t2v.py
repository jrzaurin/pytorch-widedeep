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
    TabFastFormer,
    TabTransformer,
)
from pytorch_widedeep.preprocessing import TabPreprocessor

colnames = list(string.ascii_lowercase)[:4] + ["target"]
cat_col1_vals = ["a", "b", "c"]
cat_col2_vals = ["d", "e", "f"]


def create_df():
    cat_cols = [np.array(choices(c, k=5)) for c in [cat_col1_vals, cat_col2_vals]]
    cont_cols = [np.round(np.random.rand(5), 2) for _ in range(2)]
    target = [np.random.choice(2, 5)]
    return pd.DataFrame(
        np.vstack(cat_cols + cont_cols + target).transpose(), columns=colnames
    )


df_init = create_df()
df_t2v = create_df()

embed_cols = [("a", 2), ("b", 4)]
cont_cols = ["c", "d"]
tab_preprocessor = TabPreprocessor(embed_cols=embed_cols, continuous_cols=cont_cols)
X_tab = tab_preprocessor.fit_transform(df_init)

tabmlp = TabMlp(
    column_idx=tab_preprocessor.column_idx,
    embed_input=tab_preprocessor.embeddings_input,
    continuous_cols=tab_preprocessor.continuous_cols,
    mlp_hidden_dims=[8, 4],
)

tabresnet = TabResnet(
    column_idx=tab_preprocessor.column_idx,
    embed_input=tab_preprocessor.embeddings_input,
    continuous_cols=tab_preprocessor.continuous_cols,
    blocks_dims=[8, 8, 4],
)

tabnet = TabNet(
    column_idx=tab_preprocessor.column_idx,
    embed_input=tab_preprocessor.embeddings_input,
    continuous_cols=tab_preprocessor.continuous_cols,
)


@pytest.mark.parametrize(
    "deeptabular",
    [tabmlp, tabresnet, tabnet],
)
def test_non_transformer_models(deeptabular):

    model = WideDeep(deeptabular=deeptabular)

    # Let's assume the model is trained
    t2v = Tab2Vec(model, tab_preprocessor)
    X_vec, _ = t2v.fit_transform(df_t2v, target_col="target")

    embed_dim = sum([el[2] for el in tab_preprocessor.embeddings_input])
    cont_dim = len(tab_preprocessor.continuous_cols)
    assert X_vec.shape[1] == embed_dim + cont_dim


@pytest.mark.parametrize(
    "model_name, with_cls_token, share_embeddings, embed_continuous",
    [
        ("tabtransformer", False, False, False),
        ("tabtransformer", True, False, False),
        ("tabtransformer", False, True, False),
        ("tabtransformer", True, False, True),
        ("saint", False, False, True),
        ("saint", True, False, True),
        ("saint", False, True, True),
        ("saint", True, False, True),
        (
            "tabperceiver",
            False,
            False,
            True,
        ),  # embed_continuous is irrelevant for the perceiver
        ("tabperceiver", True, False, True),
        ("tabperceiver", False, True, True),
        ("tabperceiver", True, False, True),
        ("tabfastformer", False, False, True),
        ("tabfastformer", True, False, True),
        ("tabfastformer", False, True, True),
        ("tabfastformer", True, False, True),
    ],
)
def test_transformer_models(
    model_name, with_cls_token, share_embeddings, embed_continuous
):

    embed_cols = ["a", "b"]
    cont_cols = ["c", "d"]

    tab_preprocessor = TabPreprocessor(
        embed_cols=embed_cols,
        continuous_cols=cont_cols,
        for_transformer=True,
        with_cls_token=with_cls_token,
        shared_embed=share_embeddings,
    )
    X_tab = tab_preprocessor.fit_transform(df_init)  # noqa: F841

    if model_name == "tabtransformer":
        deeptabular = TabTransformer(
            column_idx=tab_preprocessor.column_idx,
            embed_input=tab_preprocessor.embeddings_input,
            continuous_cols=tab_preprocessor.continuous_cols,
            embed_continuous=embed_continuous,
            input_dim=8,
            n_heads=2,
            n_blocks=2,
        )
    elif model_name == "saint":
        deeptabular = SAINT(
            column_idx=tab_preprocessor.column_idx,
            embed_input=tab_preprocessor.embeddings_input,
            continuous_cols=tab_preprocessor.continuous_cols,
            embed_continuous=True,
            input_dim=8,
            n_heads=2,
            n_blocks=2,
        )
    elif model_name == "tabperceiver":
        deeptabular = TabPerceiver(
            column_idx=tab_preprocessor.column_idx,
            embed_input=tab_preprocessor.embeddings_input,
            continuous_cols=tab_preprocessor.continuous_cols,
            input_dim=8,
            n_cross_attn_heads=2,
            n_latents=2,
            latent_dim=8,
            n_latent_heads=2,
            n_perceiver_blocks=2,
            share_weights=False,
        )
    elif model_name == "tabfastformer":
        deeptabular = TabFastFormer(
            column_idx=tab_preprocessor.column_idx,
            embed_input=tab_preprocessor.embeddings_input,
            continuous_cols=tab_preprocessor.continuous_cols,
            embed_continuous=embed_continuous,
            n_blocks=2,
            n_heads=4,
            share_qv_weights=False,
            share_weights=False,
        )

    # Let's assume the model is trained
    model = WideDeep(deeptabular=deeptabular)
    t2v = Tab2Vec(model, tab_preprocessor)
    X_vec = t2v.transform(df_t2v)

    if embed_continuous:
        out_dim = (len(embed_cols) + len(cont_cols)) * deeptabular.input_dim
    else:
        out_dim = len(embed_cols) * deeptabular.input_dim + len(cont_cols)

    assert X_vec.shape[1] == out_dim
