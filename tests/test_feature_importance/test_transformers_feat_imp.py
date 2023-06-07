import numpy as np
import pandas as pd
import pytest

from pytorch_widedeep import Trainer
from pytorch_widedeep.models import (
    SAINT,
    WideDeep,
    FTTransformer,
    TabFastFormer,
    TabTransformer,
    SelfAttentionMLP,
    ContextAttentionMLP,
)
from pytorch_widedeep.preprocessing import TabPreprocessor

np.random.seed(42)

# Define the column names
cat_cols = ["cat1", "cat2", "cat3", "cat4"]
cont_cols = ["cont1", "cont2", "cont3", "cont4"]
columns = cat_cols + cont_cols

# Generate random categorical data
categorical_data = np.random.choice(["A", "B", "C"], size=(32, 4))

# Generate random numerical data
numerical_data = np.random.randn(32, 4)

# Create the DataFrame
data = np.concatenate((categorical_data, numerical_data), axis=1)
df = pd.DataFrame(data, columns=columns)
target = np.random.choice(2, 32)

df_tr = df[:16].copy()
df_te = df[16:].copy().reset_index(drop=True)

y_tr = target[:16]
y_te = target[16:]

# ############################TESTS BEGIN #######################################


def _build_model_for_feat_imp_test(model_name, params):
    if model_name == "tabtransformer":
        return TabTransformer(
            input_dim=6, n_blocks=2, n_heads=2, embed_continuous=True, **params
        )
    if model_name == "saint":
        return SAINT(input_dim=6, n_blocks=2, n_heads=2, **params)
    if model_name == "fttransformer":
        return FTTransformer(
            input_dim=6, n_blocks=2, n_heads=2, kv_compression_factor=1.0, **params
        )
    if model_name == "tabfastformer":
        return TabFastFormer(input_dim=6, n_blocks=2, n_heads=2, **params)
    if model_name == "self_attn_mlp":
        return SelfAttentionMLP(input_dim=6, n_blocks=2, n_heads=2, **params)
    if model_name == "cxt_attn_mlp":
        return ContextAttentionMLP(input_dim=6, n_blocks=2, **params)


@pytest.mark.parametrize("with_cls_token", [True, False])
@pytest.mark.parametrize(
    "model_name",
    [
        "tabtransformer",
        "saint",
        "fttransformer",
        "tabfastformer",
        "self_attn_mlp",
        "cxt_attn_mlp",
    ],
)
def test_feature_importances(with_cls_token, model_name):
    tab_preprocessor = TabPreprocessor(
        cat_embed_cols=cat_cols,
        continuous_cols=cont_cols,
        with_attention=True,
        with_cls_token=with_cls_token,
    )
    X_tr = tab_preprocessor.fit_transform(df_tr).astype(float)
    X_te = tab_preprocessor.transform(df_te).astype(float)

    params = {
        "column_idx": tab_preprocessor.column_idx,
        "cat_embed_input": tab_preprocessor.cat_embed_input,
        "continuous_cols": tab_preprocessor.continuous_cols,
    }

    tab_model = _build_model_for_feat_imp_test(model_name, params)

    model = WideDeep(deeptabular=tab_model)

    trainer = Trainer(
        model,
        objective="binary",
    )

    trainer.fit(
        X_tab=X_tr,
        target=target,
        n_epochs=1,
        batch_size=16,
        feature_importance_sample_size=1000,
    )

    feat_imps = trainer.feature_importance
    feat_imp_per_sample = trainer.explain(X_te)

    assert len(feat_imps) == df_tr.shape[1] and feat_imp_per_sample.shape == df_te.shape


def test_fttransformer_valueerror():
    tab_preprocessor = TabPreprocessor(
        cat_embed_cols=cat_cols,
        continuous_cols=cont_cols,
        with_attention=True,
    )
    X_tr = tab_preprocessor.fit_transform(df_tr).astype(float)

    params = {
        "column_idx": tab_preprocessor.column_idx,
        "cat_embed_input": tab_preprocessor.cat_embed_input,
        "continuous_cols": tab_preprocessor.continuous_cols,
    }

    model = FTTransformer(
        input_dim=6, n_blocks=2, n_heads=2, kv_compression_factor=0.5, **params
    )
    model = WideDeep(deeptabular=model)

    trainer = Trainer(
        model,
        objective="binary",
    )

    with pytest.raises(ValueError) as ve:
        trainer.fit(
            X_tab=X_tr,
            target=target,
            n_epochs=1,
            batch_size=16,
            feature_importance_sample_size=1000,
        )

    assert (
        ve.value.args[0]
        == "Feature importance can only be computed if the compression factor 'kv_compression_factor' is set to 1"
    )


# TODO: Tabnet
