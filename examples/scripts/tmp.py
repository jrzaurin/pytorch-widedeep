import numpy as np
import pandas as pd

from pytorch_widedeep import Trainer
from pytorch_widedeep.models import TabMlp, WideDeep
from pytorch_widedeep.metrics import Accuracy, Precision
from pytorch_widedeep.datasets import load_adult
from pytorch_widedeep.preprocessing import TabPreprocessor

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

    tab_mlp = TabMlp(
        column_idx=tab_preprocessor.column_idx,
        cat_embed_input=tab_preprocessor.cat_embed_input,
        continuous_cols=continuous_cols,
        mlp_hidden_dims=[200, 100],
        mlp_dropout=[0.2, 0.2],
    )

    model = WideDeep(deeptabular=tab_mlp)

    trainer = Trainer(
        model,
        objective="binary",
        metrics=[Accuracy, Precision],
    )

    trainer.fit(
        X_tab=X_tab,
        target=target,
        n_epochs=2,
        batch_size=256,
        val_split=0.2,
    )
