import numpy as np
import torch
import pandas as pd

from pytorch_widedeep.models.tabular.embeddings_layers import (
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
    piecewise_linear_cont_embeddings = PiecewiseLinearContEmbeddings(
        column_idx={col: i for i, col in enumerate(df.columns)},
        quantization_setup=percentiles_dict,
        embed_dim=10,
        embed_dropout=0.1,
        use_bias=True,
    )
    out = piecewise_linear_cont_embeddings(X)
