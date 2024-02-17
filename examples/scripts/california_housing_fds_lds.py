import torch
import pandas as pd

from pytorch_widedeep import Trainer
from pytorch_widedeep.models import TabMlp, WideDeep
from pytorch_widedeep.datasets import load_california_housing
from pytorch_widedeep.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_widedeep.preprocessing import TabPreprocessor

use_cuda = torch.cuda.is_available()

if __name__ == "__main__":
    df: pd.DataFrame = load_california_housing(as_frame=True)

    target = df.MedHouseVal.values
    df = df.drop("MedHouseVal", axis=1)

    continuous_cols = df.columns.tolist()
    tab_preprocessor = TabPreprocessor(continuous_cols=continuous_cols, scale=True)
    X_tab = tab_preprocessor.fit_transform(df)

    tab_mlp = TabMlp(
        column_idx=tab_preprocessor.column_idx,
        continuous_cols=continuous_cols,
        mlp_hidden_dims=[200, 100],
        mlp_dropout=[0.2, 0.2],
    )
    model = WideDeep(deeptabular=tab_mlp, with_fds=True, enforce_positive=True)

    model_checkpoint = ModelCheckpoint(
        filepath="model_weights/wd_out",
        save_best_only=True,
        max_save=1,
    )
    early_stopping = EarlyStopping(patience=5)
    callbacks = [early_stopping, model_checkpoint]

    trainer = Trainer(
        model,
        objective="regression",
        callbacks=callbacks,
    )

    trainer.fit(
        X_tab=X_tab,
        target=target,
        n_epochs=2,
        batch_size=256,
        val_split=0.2,
        with_lds=True,
        lds_kernel="triang",
        lds_granularity=200,
    )
