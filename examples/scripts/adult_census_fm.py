import torch
import pandas as pd

from pytorch_widedeep import Trainer
from pytorch_widedeep.models import WideDeep as ModelConstructor
from pytorch_widedeep.metrics import Accuracy
from pytorch_widedeep.datasets import load_adult
from pytorch_widedeep.models.rec import (
    FactorizationMachine,
    FieldAwareFactorizationMachine,
)
from pytorch_widedeep.preprocessing import TabPreprocessor

use_cuda = torch.cuda.is_available()

if __name__ == "__main__":
    df: pd.DataFrame = load_adult(as_frame=True)
    df.columns = [c.replace("-", "_") for c in df.columns]
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
        # for_transformer=True,
        # scale=True,  # type: ignore[arg-type]
    )
    X_tab = tab_preprocessor.fit_transform(df)

    age_quantiles = (
        [df["age"].min()]
        + list(pd.qcut(df["age"], q=3, retbins=True)[1][1:-1])
        + [df["age"].max()]
    )
    hours_per_week_quantiles = (
        [df["hours_per_week"].min()]
        + list(pd.qcut(df["hours_per_week"], q=2, retbins=True)[1][1:-1])
        + [df["hours_per_week"].max()]
    )

    fm = FieldAwareFactorizationMachine(
        column_idx=tab_preprocessor.column_idx,
        cat_embed_input=tab_preprocessor.cat_embed_input,
        num_factors=4,
        continuous_cols=continuous_cols,
        embed_continuous_method="piecewise",
        quantization_setup={
            "age": age_quantiles,
            "hours_per_week": hours_per_week_quantiles,
        },
    )

    model = ModelConstructor(deeptabular=fm)

    trainer = Trainer(model, objective="binary", metrics=[Accuracy])

    trainer.fit(X_tab=X_tab, target=target, n_epochs=40, batch_size=256, val_split=0.2)
