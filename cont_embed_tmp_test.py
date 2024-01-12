import numpy as np
import torch
import pandas as pd

from pytorch_widedeep.models.tabular.embeddings_layers import (
    PeriodicLinearContEmbeddings,
    PiecewiseLinearContEmbeddings,
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

    # # PiecewiseLinearContEmbeddings
    piecewise_linear_cont_embeddings = PiecewiseLinearContEmbeddings(
        column_idx={col: i for i, col in enumerate(df.columns)},
        quantization_setup=percentiles_dict,
        embed_dim=10,
        embed_dropout=0.1,
    )
    out = piecewise_linear_cont_embeddings(X)

    # PeriodicLinearContEmbeddings
    periodic_linear_cont_embeddings = PeriodicLinearContEmbeddings(
        n_cont_cols=X.shape[1],
        embed_dim=10,
        embed_dropout=0.1,
        n_frequencies=6,
        sigma=0.01,
        share_last_layer=False,
    )
    out = periodic_linear_cont_embeddings(X)
