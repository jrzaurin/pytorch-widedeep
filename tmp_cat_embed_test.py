import numpy as np
import torch
import pandas as pd

from pytorch_widedeep.preprocessing import TabPreprocessor
from pytorch_widedeep.models.tabular.embeddings_layers import (
    SharedEmbeddings,
    DiffSizeCatEmbeddings,
    SameSizeCatEmbeddings,
)

# Define categories for each column
categories_col1 = ["A", "B", "C"]
categories_col2 = ["X", "Y", "Z"]
categories_col3 = ["Red", "Green", "Blue"]
categories_col4 = ["Small", "Medium", "Large"]

# Generate random data
data = {
    "Category1": np.random.choice(categories_col1, size=32),
    "Category2": np.random.choice(categories_col2, size=32),
    "Category3": np.random.choice(categories_col3, size=32),
    "Category4": np.random.choice(categories_col4, size=32),
    "target": np.random.randint(2, size=32),
}

# Create DataFrame
df = pd.DataFrame(data)

shared_embeddings_processor = TabPreprocessor(
    cat_embed_cols=[c for c in df.columns if "Category" in c],
    for_transformer=True,
    shared_embed=True,
)

X_tab = shared_embeddings_processor.fit_transform(df)
X_tab = torch.tensor(X_tab, dtype=torch.long)

shared_embeddings = torch.nn.ModuleDict(
    {
        "emb_layer_"
        + col: SharedEmbeddings(
            val + 1,
            embed_dim=6,
        )
        for col, val in shared_embeddings_processor.cat_embed_input
    }
)

shared_embeddings_cat_embed = [
    shared_embeddings["emb_layer_" + col](
        X_tab[:, shared_embeddings_processor.column_idx[col]]
    ).unsqueeze(1)
    for col, _ in shared_embeddings_processor.cat_embed_input
]
x = torch.cat(shared_embeddings_cat_embed, 1)

diff_size_cat_embeddings_processor = TabPreprocessor(
    cat_embed_cols=[c for c in df.columns if "Category" in c],
)

X_tab = diff_size_cat_embeddings_processor.fit_transform(df)
X_tab = torch.tensor(X_tab, dtype=torch.long)

diff_size_cat_embed = DiffSizeCatEmbeddings(
    column_idx=diff_size_cat_embeddings_processor.column_idx,
    embed_input=diff_size_cat_embeddings_processor.cat_embed_input,
    embed_dropout=0.1,
    use_bias=True,
    activation_fn="relu",
)

x = diff_size_cat_embed(X_tab)

same_size_cat_embeddings_processor = TabPreprocessor(
    cat_embed_cols=[c for c in df.columns if "Category" in c],
    for_transformer=True,
    shared_embed=False,
)

X_tab = same_size_cat_embeddings_processor.fit_transform(df)
X_tab = torch.tensor(X_tab, dtype=torch.long)

same_size_cat_embed = SameSizeCatEmbeddings(
    embed_dim=6,
    column_idx=same_size_cat_embeddings_processor.column_idx,
    embed_input=same_size_cat_embeddings_processor.cat_embed_input,
    embed_dropout=0.1,
    use_bias=True,
    full_embed_dropout=False,
    shared_embed=False,
    add_shared_embed=False,
    frac_shared_embed=0.5,  # need to make this optional
    activation_fn="relu",
)

x = same_size_cat_embed(X_tab)
