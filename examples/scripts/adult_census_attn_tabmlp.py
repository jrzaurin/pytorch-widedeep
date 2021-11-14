from pathlib import Path

import numpy as np
import torch
import pandas as pd

from pytorch_widedeep import Trainer
from pytorch_widedeep.models import WideDeep, AttentiveTabMlp
from pytorch_widedeep.metrics import Accuracy
from pytorch_widedeep.callbacks import EarlyStopping
from pytorch_widedeep.preprocessing import TabPreprocessor

use_cuda = torch.cuda.is_available()

if __name__ == "__main__":

    DATA_PATH = Path("../data")

    df = pd.read_csv(DATA_PATH / "adult/adult.csv.zip")
    df.columns = [c.replace("-", "_") for c in df.columns]
    df["age_buckets"] = pd.cut(
        df.age, bins=[16, 25, 30, 35, 40, 45, 50, 55, 60, 91], labels=np.arange(9)
    )
    df["income_label"] = (df["income"].apply(lambda x: ">50K" in x)).astype(int)
    df.drop("income", axis=1, inplace=True)
    df.head()

    cat_embed_cols = [
        "education",
        "relationship",
        "workclass",
        "occupation",
        "native_country",
    ]
    continuous_cols = ["age", "hours_per_week"]
    target = "income_label"
    target = df[target].values

    tab_preprocessor = TabPreprocessor(
        embed_cols=cat_embed_cols, continuous_cols=continuous_cols, with_attention=True
    )
    X_tab = tab_preprocessor.fit_transform(df)

    tab_mlp_attn = AttentiveTabMlp(
        mlp_hidden_dims=[200, 100],
        mlp_dropout=0.1,
        column_idx=tab_preprocessor.column_idx,
        cat_embed_input=tab_preprocessor.embeddings_input,
        continuous_cols=continuous_cols,
        attention_name="context_attention",
    )
    model = WideDeep(deeptabular=tab_mlp_attn)

    trainer = Trainer(
        model,
        objective="binary",
        callbacks=[EarlyStopping(patience=5)],
        metrics=[Accuracy],
    )
    trainer.fit(
        X_tab=X_tab,
        target=target,
        n_epochs=2,
        batch_size=256,
        val_split=0.2,
    )
