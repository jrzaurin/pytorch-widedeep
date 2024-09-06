import numpy as np
import torch
import pandas as pd

from pytorch_widedeep import Trainer
from pytorch_widedeep.models import WideDeep as ModelConstructor
from pytorch_widedeep.metrics import Accuracy
from pytorch_widedeep.datasets import load_adult
from pytorch_widedeep.models.rec import FactorizationMachine
from pytorch_widedeep.preprocessing import TabPreprocessor

use_cuda = torch.cuda.is_available()

if __name__ == "__main__":
    df: pd.DataFrame = load_adult(as_frame=True)
    df.columns = [c.replace("-", "_") for c in df.columns]
    df["age_buckets"] = pd.cut(
        df.age, bins=[16, 25, 30, 35, 40, 45, 50, 55, 60, 91], labels=np.arange(9)
    )
    df["income_label"] = (df["income"].apply(lambda x: ">50K" in x)).astype(int)
    df.drop("income", axis=1, inplace=True)

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
        cat_embed_cols=cat_embed_cols,
        continuous_cols=continuous_cols,
        scale=True,  # type: ignore[arg-type]
    )
    X_tab = tab_preprocessor.fit_transform(df)

    fm = FactorizationMachine(
        column_idx=tab_preprocessor.column_idx,
        num_factors=4,
        cat_embed_input=tab_preprocessor.cat_embed_input,
    )

    model = ModelConstructor(deeptabular=fm)

    trainer = Trainer(model, objective="binary", metrics=[Accuracy])

    trainer.fit(X_tab=X_tab, target=target, n_epochs=40, batch_size=256, val_split=0.2)
