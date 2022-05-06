import numpy as np
import torch
import pandas as pd

from pytorch_widedeep.models import TabTransformer
from pytorch_widedeep.datasets import load_adult
from pytorch_widedeep.preprocessing import TabPreprocessor
from pytorch_widedeep.self_supervised_training.self_supervised_trainer import (
    SelfSupervisedTrainer,
)

use_cuda = torch.cuda.is_available()

if __name__ == "__main__":

    df = load_adult(as_frame=True)
    df.columns = [c.replace("-", "_") for c in df.columns]
    df["age_buckets"] = pd.cut(
        df.age, bins=[16, 25, 30, 35, 40, 45, 50, 55, 60, 91], labels=np.arange(9)
    )
    df["income_label"] = (df["income"].apply(lambda x: ">50K" in x)).astype(int)
    df.drop("income", axis=1, inplace=True)

    # Define wide, crossed and deep tabular columns
    wide_cols = [
        "age_buckets",
        "education",
        "relationship",
        "workclass",
        "occupation",
        "native_country",
        "gender",
    ]
    crossed_cols = [("education", "occupation"), ("native_country", "occupation")]

    cat_embed_cols = [
        "workclass",
        "education",
        "marital_status",
        "occupation",
        "relationship",
        "race",
        "gender",
        "capital_gain",
        "capital_loss",
        "native_country",
    ]
    continuous_cols = ["age", "hours_per_week"]
    target = "income_label"
    target = df[target].values

    tab_preprocessor = TabPreprocessor(
        cat_embed_cols=cat_embed_cols, continuous_cols=continuous_cols  # type: ignore[arg-type]
    )
    X_tab = tab_preprocessor.fit_transform(df)

    tab_mlp = TabTransformer(
        column_idx=tab_preprocessor.column_idx,
        cat_embed_input=tab_preprocessor.cat_embed_input,
        continuous_cols=continuous_cols,
        embed_continuous=True,
        mlp_hidden_dims=[200, 100],
        mlp_dropout=0.2,
    )

    ss_trainer = SelfSupervisedTrainer(tab_mlp)
    ss_trainer.pretrain(X_tab, n_epochs=1, batch_size=256)
