# Some embedding methods are tested somewhere within the test files. Here I
# will focus on the new embedding methods included for continuous features

import numpy as np
import torch
import pandas as pd
import pytest

from pytorch_widedeep.models import TabMlp, TabTransformer
from pytorch_widedeep.preprocessing import TabPreprocessor
from pytorch_widedeep.models.tabular.embeddings_layers import (
    ContEmbeddings,
    PeriodicContEmbeddings,
    PiecewiseContEmbeddings,
)

data_size = 32
embed_dim = 8

# Create categorical data
categories = ["A", "B", "C", "D"]
cat_data = np.random.choice(categories, size=data_size)

# Create numerical data
num_data_1 = np.random.rand(data_size)
num_data_2 = np.random.rand(data_size)

# Create DataFrame
df = pd.DataFrame(
    {
        "cat1": cat_data,
        "cat2": cat_data[::-1],  # Reverse of cat_data
        "num1": num_data_1,
        "num2": num_data_2,
    }
)

percentiles_dict = {
    col: [
        df[col].min(),
        df[col].quantile(0.25),
        df[col].quantile(0.5),
        df[col].quantile(0.75),
        df[col].max(),
    ]
    for col in df.columns
    if "num" in col
}


tab_preprocessor = TabPreprocessor(
    cat_embed_cols=["cat1", "cat2"], continuous_cols=["num1", "num2"]
)
X_tab = tab_preprocessor.fit_transform(df)
X = torch.Tensor(X_tab)

standard_params = {
    "n_cont_cols": 2,
    "embed_dim": embed_dim,
    "embed_dropout": 0.1,
    "full_embed_dropout": False,
}

periodic_params = {
    "n_cont_cols": 2,
    "embed_dim": embed_dim,
    "embed_dropout": 0.1,
    "full_embed_dropout": False,
    "n_frequencies": 4,
    "sigma": 0.1,
    "share_last_layer": False,
}

piecewise_params = {
    "column_idx": {"num1": 0, "num2": 1},
    "quantization_setup": percentiles_dict,
    "embed_dim": embed_dim,
    "embed_dropout": 0.1,
    "full_embed_dropout": False,
}


def _build_num_embedder(embedding_method, extra_params=None):
    if embedding_method == "standard":
        standard_params.update(extra_params or {})
        return ContEmbeddings(**standard_params)
    elif embedding_method == "periodic":
        periodic_params.update(extra_params or {})
        return PeriodicContEmbeddings(**periodic_params)
    elif embedding_method == "piecewise":
        piecewise_params.update(extra_params or {})
        return PiecewiseContEmbeddings(**piecewise_params)


@pytest.mark.parametrize("method", ["standard", "periodic", "piecewise"])
def test_cont_embeddings_methods_directly(method):
    num_embedder = _build_num_embedder(method)
    cont_idx = [2, 3]
    out = num_embedder(X[:, cont_idx])
    assert out.shape == (X.shape[0], len(cont_idx), embed_dim)


@pytest.mark.parametrize("method", ["standard", "periodic", "piecewise"])
def test_cont_embeddings_methods_tabmlp(method):
    mlp_hidden_dims = [16, 8]
    tab_mlp = TabMlp(
        column_idx=tab_preprocessor.column_idx,
        cat_embed_input=tab_preprocessor.cat_embed_input,
        continuous_cols=tab_preprocessor.continuous_cols,
        embed_continuous_method=method,
        cont_embed_activation="relu",
        quantization_setup=percentiles_dict if method == "piecewise" else None,
        cont_embed_dim=embed_dim,
        n_frequencies=4 if method == "periodic" else None,
        sigma=0.3 if method == "periodic" else None,
        share_last_layer=True if method == "periodic" else None,
        mlp_hidden_dims=mlp_hidden_dims,
    )
    out = tab_mlp(X)
    assert out.shape == (X.shape[0], mlp_hidden_dims[-1])


@pytest.mark.parametrize("method", ["standard", "periodic", "piecewise"])
def test_cont_embeddings_methods_tabtransformer(method):
    tab_transformer = TabTransformer(
        column_idx=tab_preprocessor.column_idx,
        cat_embed_input=tab_preprocessor.cat_embed_input,
        continuous_cols=tab_preprocessor.continuous_cols,
        embed_continuous_method=method,
        cont_embed_activation="relu",
        quantization_setup=percentiles_dict if method == "piecewise" else None,
        n_frequencies=4 if method == "periodic" else None,
        sigma=0.3 if method == "periodic" else None,
        share_last_layer=True if method == "periodic" else None,
        input_dim=8,
    )
    out = tab_transformer(X)
    assert out.shape == (X.shape[0], embed_dim * X.shape[1])


@pytest.mark.parametrize("method", ["standard", "periodic", "piecewise"])
def test_full_dropout(method):
    # Create DataFrame
    df = pd.DataFrame(
        {
            "num1": np.random.rand(data_size),
            "num2": np.random.rand(data_size),
            "num3": np.random.rand(data_size),
            "num4": np.random.rand(data_size),
            "num5": np.random.rand(data_size),
        }
    )

    percentiles_dict = {
        col: [
            df[col].min(),
            df[col].quantile(0.25),
            df[col].quantile(0.5),
            df[col].quantile(0.75),
            df[col].max(),
        ]
        for col in df.columns
    }
    if method == "standard":
        num_embedder = _build_num_embedder(
            "standard",
            {"n_cont_cols": 5, "full_embed_dropout": True, "embed_dropout": 0.8},
        )
    if method == "periodic":
        num_embedder = _build_num_embedder(
            "periodic",
            {
                "n_cont_cols": 5,
                "full_embed_dropout": True,
                "embed_dropout": 0.8,
            },
        )
    if method == "piecewise":
        num_embedder = _build_num_embedder(
            "piecewise",
            {
                "column_idx": {"num1": 0, "num2": 1, "num3": 2, "num4": 3, "num5": 4},
                "quantization_setup": percentiles_dict,
                "full_embed_dropout": True,
                "embed_dropout": 0.8,
            },
        )

    X = torch.Tensor(df.values)
    out = num_embedder(X)

    is_a_column_all_zeros = []
    for i in range(out.shape[1]):
        is_a_column_all_zeros.append(out[:, i, :].sum().item() == 0.0)

    assert any(is_a_column_all_zeros)
