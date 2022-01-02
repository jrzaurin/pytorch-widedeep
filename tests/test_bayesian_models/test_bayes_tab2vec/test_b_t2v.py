import string
from random import choices

import numpy as np
import pandas as pd
import pytest

from pytorch_widedeep import Tab2Vec
from pytorch_widedeep.preprocessing import TabPreprocessor
from pytorch_widedeep.bayesian_models import BayesianTabMlp

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


@pytest.mark.parametrize("return_dataframe", [True, False])
@pytest.mark.parametrize("embed_continuous", [True, False])
def test_bayesian_mlp_models(return_dataframe, embed_continuous):

    tab_preprocessor = TabPreprocessor(
        cat_embed_cols=embed_cols, continuous_cols=cont_cols
    )
    X_tab = tab_preprocessor.fit_transform(df_init)  # noqa: F841

    model = BayesianTabMlp(
        column_idx=tab_preprocessor.column_idx,
        cat_embed_input=tab_preprocessor.embeddings_input,
        continuous_cols=tab_preprocessor.continuous_cols,
        embed_continuous=embed_continuous,
        mlp_hidden_dims=[8, 4],
    )

    # Let's assume the model is trained
    t2v = Tab2Vec(model, tab_preprocessor, return_dataframe=return_dataframe)
    t2v_out, _ = t2v.fit_transform(df_t2v, target_col="target")

    embed_dim = sum([el[2] for el in tab_preprocessor.embeddings_input])
    n_cont_cols = len(tab_preprocessor.continuous_cols)
    cont_dim = n_cont_cols * model.cont_embed_dim if embed_continuous else n_cont_cols

    assert t2v_out.shape[1] == embed_dim + cont_dim
