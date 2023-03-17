import numpy as np
import torch
import pandas as pd

from pytorch_widedeep import Trainer
from pytorch_widedeep.models import TabNet, WideDeep
from pytorch_widedeep.metrics import Accuracy, Precision
from pytorch_widedeep.datasets import load_adult
from pytorch_widedeep.callbacks import (
    LRHistory,
    EarlyStopping,
    ModelCheckpoint,
)
from pytorch_widedeep.preprocessing import TabPreprocessor

use_cuda = torch.cuda.is_available()

if __name__ == "__main__":

    df = load_adult(as_frame=True)
    df.columns = [c.replace("-", "_") for c in df.columns]
    df["age_buckets"] = pd.cut(
        df.age, bins=[16, 25, 30, 35, 40, 45, 50, 55, 60, 91], labels=np.arange(9)
    )
    df["income_label"] = (df["income"].apply(lambda x: ">50K" in x)).astype(int)
    df.drop("income", axis=1, inplace=True)
    df.head()

    cat_embed_cols = [
        "age_buckets",
        "education",
        "relationship",
        "workclass",
        "occupation",
        "native_country",
        "gender",
    ]
    continuous_cols = ["age", "hours_per_week"]

    target = "income_label"
    target = df[target].values

    tab_preprocessor = TabPreprocessor(
        cat_embed_cols=cat_embed_cols,
        continuous_cols=continuous_cols,
        scale=True,
        default_embed_dim=1,
    )
    X_tab = tab_preprocessor.fit_transform(df)

    deeptabular = TabNet(
        column_idx=tab_preprocessor.column_idx,
        cat_embed_input=tab_preprocessor.cat_embed_input,
        continuous_cols=continuous_cols,
        n_steps=3,
        attn_dim=32,
        ghost_bn=False,
    )

    model = WideDeep(deeptabular=deeptabular)

    callbacks = [
        LRHistory(n_epochs=10),
        EarlyStopping(patience=5),
        ModelCheckpoint(filepath="model_weights/wd_out"),
    ]
    metrics = [Accuracy, Precision]

    trainer = Trainer(
        model,
        objective="binary",
        optimizers=torch.optim.Adam(model.parameters(), lr=0.01),
        callbacks=callbacks,
        metrics=metrics,
    )

    trainer.fit(
        X_tab=X_tab,
        target=target,
        n_epochs=2,
        batch_size=256,
        val_split=0.2,
    )
