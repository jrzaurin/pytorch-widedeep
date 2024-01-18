import numpy as np
import torch
import pandas as pd

from pytorch_widedeep.preprocessing import TabPreprocessor
from pytorch_widedeep.models.tabular._base_tabular_model import (
    BaseTabularModelWithAttention,
    BaseTabularModelWithoutAttention,
)

if __name__ == "__main__":
    # Define categories for each column
    categories_col1 = ["A", "B", "C"]
    categories_col2 = ["X", "Y", "Z"]
    categories_col3 = ["Red", "Green", "Blue"]
    categories_col4 = ["Small", "Medium", "Large"]

    # Generate random data
    data = {
        "cont_1": np.random.randn(32),
        "cont_2": np.random.randn(32),
        "cont_3": np.random.randn(32),
        "cont_4": np.random.randn(32),
        "catg_1": np.random.choice(categories_col1, size=32),
        "catg_2": np.random.choice(categories_col2, size=32),
        "catg_3": np.random.choice(categories_col3, size=32),
        "catg_4": np.random.choice(categories_col4, size=32),
        "target": np.random.randint(2, size=32),
    }

    # Create DataFrame
    df = pd.DataFrame(data)

    percentiles_dict = {
        col: [
            df[col].min(),
            df[col].quantile(0.25),
            df[col].quantile(0.5),
            df[col].quantile(0.75),
            df[col].max(),
        ]
        for col in df.columns
        if "cont" in col
    }

    tab_preprocessor = TabPreprocessor(
        cat_embed_cols=[c for c in df.columns if "catg" in c],
        continuous_cols=[c for c in df.columns if "cont" in c],
    )

    X_tab = tab_preprocessor.fit_transform(df)
    X_tab_tnsr = torch.tensor(X_tab)

    # BaseTabularModelWithoutAttention
    model_no_cont_embed = BaseTabularModelWithoutAttention(
        column_idx=tab_preprocessor.column_idx,
        cat_embed_input=tab_preprocessor.cat_embed_input,
        cat_embed_dropout=0.1,
        use_cat_bias=True,
        cat_embed_activation="relu",
        continuous_cols=tab_preprocessor.continuous_cols,
        cont_norm_layer="batchnorm",
        embed_continuous=False,
        embed_continuous_method=None,
        cont_embed_dim=12,
        cont_embed_dropout=0.1,
        cont_embed_activation="relu",
        quantization_setup=percentiles_dict,
        n_frequencies=4,
        sigma=0.1,
        share_last_layer=False,
    )

    out = model_no_cont_embed._get_embeddings(X_tab_tnsr)

    model_cont_embed = BaseTabularModelWithoutAttention(
        column_idx=tab_preprocessor.column_idx,
        cat_embed_input=tab_preprocessor.cat_embed_input,
        cat_embed_dropout=0.1,
        use_cat_bias=True,
        cat_embed_activation="relu",
        continuous_cols=tab_preprocessor.continuous_cols,
        cont_norm_layer="batchnorm",
        embed_continuous=True,
        embed_continuous_method="standard",
        cont_embed_dim=12,
        cont_embed_dropout=0.1,
        cont_embed_activation="relu",
        quantization_setup=percentiles_dict,
        n_frequencies=4,
        sigma=0.1,
        share_last_layer=False,
    )

    out = model_cont_embed._get_embeddings(X_tab_tnsr)

    model_cont_piece_embed = BaseTabularModelWithoutAttention(
        column_idx=tab_preprocessor.column_idx,
        cat_embed_input=tab_preprocessor.cat_embed_input,
        cat_embed_dropout=0.1,
        use_cat_bias=True,
        cat_embed_activation="relu",
        continuous_cols=tab_preprocessor.continuous_cols,
        cont_norm_layer="batchnorm",
        embed_continuous=True,
        embed_continuous_method="piecewise",
        cont_embed_dim=12,
        cont_embed_dropout=0.1,
        cont_embed_activation="relu",
        quantization_setup=percentiles_dict,
        n_frequencies=4,
        sigma=0.1,
        share_last_layer=False,
    )

    out = model_cont_piece_embed._get_embeddings(X_tab_tnsr)

    model_cont_periodic_embed = BaseTabularModelWithoutAttention(
        column_idx=tab_preprocessor.column_idx,
        cat_embed_input=tab_preprocessor.cat_embed_input,
        cat_embed_dropout=0.1,
        use_cat_bias=True,
        cat_embed_activation="relu",
        continuous_cols=tab_preprocessor.continuous_cols,
        cont_norm_layer="batchnorm",
        embed_continuous=True,
        embed_continuous_method="periodic",
        cont_embed_dim=12,
        cont_embed_dropout=0.1,
        cont_embed_activation="relu",
        quantization_setup=percentiles_dict,
        n_frequencies=4,
        sigma=0.1,
        share_last_layer=False,
    )

    out = model_cont_periodic_embed._get_embeddings(X_tab_tnsr)

    tab_preprocessor = TabPreprocessor(
        cat_embed_cols=[c for c in df.columns if "catg" in c],
        continuous_cols=[c for c in df.columns if "cont" in c],
        with_attention=True,
    )

    X_tab = tab_preprocessor.fit_transform(df)
    X_tab_tnsr = torch.tensor(X_tab)

    # BaseTabularModelWithAttention
    model_cont_embed_attn = BaseTabularModelWithAttention(
        column_idx=tab_preprocessor.column_idx,
        cat_embed_input=tab_preprocessor.cat_embed_input,
        cat_embed_dropout=0.1,
        use_cat_bias=True,
        cat_embed_activation="relu",
        full_embed_dropout=False,
        shared_embed=False,
        add_shared_embed=False,
        frac_shared_embed=0.5,
        continuous_cols=tab_preprocessor.continuous_cols,
        cont_norm_layer="batchnorm",
        embed_continuous=True,
        embed_continuous_method="standard",
        cont_embed_dropout=0.1,
        cont_embed_activation="relu",
        input_dim=12,
        quantization_setup=percentiles_dict,
        n_frequencies=4,
        sigma=0.1,
        share_last_layer=False,
    )

    out = model_cont_embed_attn._get_embeddings(X_tab_tnsr)

    model_cont_piece_embed_attn = BaseTabularModelWithAttention(
        column_idx=tab_preprocessor.column_idx,
        cat_embed_input=tab_preprocessor.cat_embed_input,
        cat_embed_dropout=0.1,
        use_cat_bias=True,
        cat_embed_activation="relu",
        full_embed_dropout=False,
        shared_embed=False,
        add_shared_embed=False,
        frac_shared_embed=0.5,
        continuous_cols=tab_preprocessor.continuous_cols,
        cont_norm_layer="batchnorm",
        embed_continuous=True,
        embed_continuous_method="piecewise",
        cont_embed_dropout=0.1,
        cont_embed_activation="relu",
        input_dim=12,
        quantization_setup=percentiles_dict,
        n_frequencies=4,
        sigma=0.1,
        share_last_layer=False,
    )

    out = model_cont_piece_embed_attn._get_embeddings(X_tab_tnsr)

    model_cont_periodic_embed_attn = BaseTabularModelWithAttention(
        column_idx=tab_preprocessor.column_idx,
        cat_embed_input=tab_preprocessor.cat_embed_input,
        cat_embed_dropout=0.1,
        use_cat_bias=True,
        cat_embed_activation="relu",
        full_embed_dropout=False,
        shared_embed=False,
        add_shared_embed=False,
        frac_shared_embed=0.5,
        continuous_cols=tab_preprocessor.continuous_cols,
        cont_norm_layer="batchnorm",
        embed_continuous=True,
        embed_continuous_method="periodic",
        cont_embed_dropout=0.1,
        cont_embed_activation="relu",
        input_dim=12,
        quantization_setup=percentiles_dict,
        n_frequencies=4,
        sigma=0.1,
        share_last_layer=False,
    )
    out = model_cont_periodic_embed_attn._get_embeddings(X_tab_tnsr)
