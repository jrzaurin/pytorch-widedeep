import warnings

import pandas as pd
from sklearn.model_selection import train_test_split

from pytorch_widedeep import Trainer
from pytorch_widedeep.models import TabMlp, WideDeep
from pytorch_widedeep.metrics import Accuracy, Precision
from pytorch_widedeep.datasets import load_bio_kdd04
from pytorch_widedeep.dataloaders import DataLoaderImbalanced
from pytorch_widedeep.preprocessing import TabPreprocessor

warnings.filterwarnings("ignore", category=DeprecationWarning)

if __name__ == "__main__":
    df: pd.DataFrame = load_bio_kdd04(as_frame=True)
    # drop columns we won't need in this example
    df.drop(columns=["EXAMPLE_ID", "BLOCK_ID"], inplace=True)

    df_train, df_valid = train_test_split(
        df, test_size=0.2, stratify=df["target"], random_state=1
    )
    df_valid, df_test = train_test_split(
        df_valid, test_size=0.5, stratify=df_valid["target"], random_state=1
    )

    continuous_cols = df.drop(columns=["target"]).columns.values.tolist()

    # deeptabular
    tab_preprocessor = TabPreprocessor(continuous_cols=continuous_cols, scale=True)
    X_tab_train = tab_preprocessor.fit_transform(df_train)
    X_tab_valid = tab_preprocessor.transform(df_valid)
    X_tab_test = tab_preprocessor.transform(df_test)

    # target
    y_train = df_train["target"].values
    y_valid = df_valid["target"].values
    y_test = df_test["target"].values

    deeptabular = TabMlp(
        column_idx=tab_preprocessor.column_idx,
        continuous_cols=tab_preprocessor.continuous_cols,
        mlp_hidden_dims=[64, 32],
    )
    model = WideDeep(deeptabular=deeptabular)

    trainer = Trainer(
        model,
        objective="binary",
        metrics=[Accuracy, Precision],
        verbose=1,
    )

    train_dataloader = DataLoaderImbalanced(kwargs={"oversample_mul": 5})

    # in reality one should not really do this, but just to illustrate
    eval_dataloader = DataLoaderImbalanced(kwargs={"oversample_mul": 5})

    trainer.fit(
        X_train={"X_tab": X_tab_train, "target": y_train},
        X_val={"X_tab": X_tab_valid, "target": y_valid},
        n_epochs=1,
        batch_size=32,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
    )

    preds = trainer.predict(X_tab=X_tab_test)
