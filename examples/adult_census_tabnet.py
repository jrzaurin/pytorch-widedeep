import numpy as np
import torch
import pandas as pd

from pytorch_widedeep import Trainer
from pytorch_widedeep.optim import RAdam
from pytorch_widedeep.models import Wide, TabNet, WideDeep
from pytorch_widedeep.metrics import Accuracy, Precision
from pytorch_widedeep.callbacks import (
    LRHistory,
    EarlyStopping,
    ModelCheckpoint,
)
from pytorch_widedeep.preprocessing import TabPreprocessor, WidePreprocessor

use_cuda = torch.cuda.is_available()

if __name__ == "__main__":

    df = pd.read_csv("data/adult/adult.csv.zip")
    df.columns = [c.replace("-", "_") for c in df.columns]
    df["age_buckets"] = pd.cut(
        df.age, bins=[16, 25, 30, 35, 40, 45, 50, 55, 60, 91], labels=np.arange(9)
    )
    df["income_label"] = (df["income"].apply(lambda x: ">50K" in x)).astype(int)
    df.drop("income", axis=1, inplace=True)
    df.head()

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
        ("education", 6),
        ("relationship", 6),
        ("workclass", 6),
        ("occupation", 6),
        ("native_country", 6),
    ]
    continuous_cols = ["age", "hours_per_week"]

    target = "income_label"
    target = df[target].values

    prepare_wide = WidePreprocessor(wide_cols=wide_cols, crossed_cols=crossed_cols)
    X_wide = prepare_wide.fit_transform(df)

    prepare_tab = TabPreprocessor(
        embed_cols=cat_embed_cols, continuous_cols=continuous_cols, default_embed_dim=1  # type: ignore[arg-type]
    )
    X_tab = prepare_tab.fit_transform(df)

    wide = Wide(wide_dim=np.unique(X_wide).shape[0], pred_dim=1)

    deeptabular = TabNet(
        column_idx=prepare_tab.column_idx,
        embed_input=prepare_tab.embeddings_input,
        continuous_cols=continuous_cols,
    )

    model = WideDeep(wide=wide, deeptabular=deeptabular)

    wide_opt = torch.optim.Adam(model.wide.parameters(), lr=0.01)
    deep_opt = RAdam(model.deeptabular.parameters(), lr=0.1)

    optimizers = {"wide": wide_opt, "deeptabular": deep_opt}
    callbacks = [
        LRHistory(n_epochs=10),
        EarlyStopping(patience=5),
        ModelCheckpoint(filepath="model_weights/wd_out"),
    ]
    metrics = [Accuracy, Precision]

    trainer = Trainer(
        model,
        objective="binary",
        optimizers=optimizers,
        callbacks=callbacks,
        metrics=metrics,
    )

    trainer.fit(
        X_wide=X_wide,
        X_tab=X_tab,
        target=target,
        n_epochs=2,
        batch_size=512,
        val_split=0.2,
    )
