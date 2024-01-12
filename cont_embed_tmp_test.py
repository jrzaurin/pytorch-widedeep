import numpy as np
import torch
import pandas as pd

from pytorch_widedeep.models.tabular.embeddings_layers import (
    ContEmbeddings,
    PeriodicContEmbeddings,
    PiecewiseContEmbeddings,
)

if __name__ == "__main__":
    np.random.seed(42)

    data = {
        "Column_A": np.random.randn(100),
        "Column_B": np.random.randn(100),
        "Column_C": np.random.randn(100),
        "Column_D": np.random.randn(100),
    }

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
    }

    X = torch.tensor(df.values, dtype=torch.float32)

    # ContEmbeddings
    cont_embeddings = ContEmbeddings(
        n_cont_cols=X.shape[1],
        embed_dim=10,
        embed_dropout=0.1,
        activation_fn="relu",
    )
    out = cont_embeddings(X)

    # # PiecewiseContEmbeddings
    piecewise_cont_embeddings = PiecewiseContEmbeddings(
        column_idx={col: i for i, col in enumerate(df.columns)},
        quantization_setup=percentiles_dict,
        embed_dim=10,
        embed_dropout=0.1,
        activation_fn="relu",

    )
    out = piecewise_cont_embeddings(X)

    # PeriodicContEmbeddings
    periodic_cont_embeddings = PeriodicContEmbeddings(
        n_cont_cols=X.shape[1],
        embed_dim=10,
        embed_dropout=0.1,
        n_frequencies=6,
        sigma=0.01,
        share_last_layer=False,
        activation_fn="relu",
    )
    out = periodic_cont_embeddings(X)
