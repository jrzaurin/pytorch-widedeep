import numpy as np
import torch
import pandas as pd

from pytorch_widedeep import Trainer
from pytorch_widedeep.models import Wide, TabMlp, WideDeep, TabResnet  # noqa: F401
from pytorch_widedeep.metrics import Accuracy, Precision
from pytorch_widedeep.datasets import load_adult
from pytorch_widedeep.callbacks import LRHistory, EarlyStopping, ModelCheckpoint
from pytorch_widedeep.initializers import XavierNormal, KaimingNormal
from pytorch_widedeep.preprocessing import TabPreprocessor, WidePreprocessor

use_cuda = torch.cuda.is_available()

if __name__ == "__main__":
    df: pd.DataFrame = load_adult(as_frame=True)
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
    # # Aternatively one could pass a list of Tuples with the name of the column
    # # and the embedding dim per column
    # cat_embed_cols = [
    #     ("education", 10),
    #     ("relationship", 8),
    #     ("workclass", 10),
    #     ("occupation", 10),
    #     ("native_country", 10),
    # ]
    target = "income_label"
    target = df[target].values

    wide_preprocessor = WidePreprocessor(wide_cols=wide_cols, crossed_cols=crossed_cols)
    X_wide = wide_preprocessor.fit_transform(df)

    tab_preprocessor = TabPreprocessor(
        cat_embed_cols=cat_embed_cols,
        continuous_cols=continuous_cols,
        scale=True,  # type: ignore[arg-type]
    )
    X_tab = tab_preprocessor.fit_transform(df)

    wide = Wide(input_dim=np.unique(X_wide).shape[0], pred_dim=1)

    # for tabular data we could use a "simple" MLP...
    tab_mlp = TabMlp(
        column_idx=tab_preprocessor.column_idx,
        cat_embed_input=tab_preprocessor.cat_embed_input,
        continuous_cols=continuous_cols,
        mlp_hidden_dims=[200, 100],
        mlp_dropout=[0.2, 0.2],
    )

    # ...or a Resnet adjusted for tabular data
    tab_resnet = TabResnet(
        column_idx=tab_preprocessor.column_idx,
        cat_embed_input=tab_preprocessor.cat_embed_input,
        continuous_cols=continuous_cols,
        blocks_dims=[200, 100],
    )

    for tab_model in [tab_mlp, tab_resnet]:
        model = WideDeep(wide=wide, deeptabular=tab_model)  # type: ignore[arg-type]

        wide_opt = torch.optim.Adam(model.wide.parameters(), lr=0.01)
        deep_opt = torch.optim.AdamW(model.deeptabular.parameters(), lr=0.01)
        wide_sch = torch.optim.lr_scheduler.StepLR(wide_opt, step_size=2)
        deep_sch = torch.optim.lr_scheduler.StepLR(deep_opt, step_size=3)

        model_checkpoint = ModelCheckpoint(
            filepath="model_weights/wd_out",
            save_best_only=True,
            max_save=1,
        )
        early_stopping = EarlyStopping(patience=5)
        optimizers = {"wide": wide_opt, "deeptabular": deep_opt}
        schedulers = {"wide": wide_sch, "deeptabular": deep_sch}
        initializers = {"wide": KaimingNormal, "deeptabular": XavierNormal}
        callbacks = [early_stopping, model_checkpoint, LRHistory(n_epochs=10)]
        metrics = [Accuracy, Precision]

        trainer = Trainer(
            model,
            objective="binary",
            optimizers=optimizers,
            lr_schedulers=schedulers,
            initializers=initializers,
            callbacks=callbacks,
            metrics=metrics,
        )

        trainer.fit(
            X_wide=X_wide,
            X_tab=X_tab,
            target=target,
            n_epochs=2,
            batch_size=256,
            val_split=0.2,
        )

        # Save and load

        # Option 1: this will also save training history and lr history if the
        # LRHistory callback is used

        # Day 0, you have trained your model, save it using the trainer.save
        # method
        trainer.save(path="model_weights", save_state_dict=True)

        # Option 2: save as any other torch model

        # Day 0, you have trained your model, save as any other torch model
        torch.save(model.state_dict(), "model_weights/wd_model.pt")

        # From here in advace, Option 1 or 2 are the same

        # Few days have passed...I assume the user has prepared the data and
        # defined the components:
        # 1. Build the model
        model_new = WideDeep(wide=wide, deeptabular=tab_model)  # type: ignore[arg-type]
        model_new.load_state_dict(torch.load("model_weights/wd_model.pt"))

        # 2. Instantiate the trainer
        trainer_new = Trainer(
            model_new,
            objective="binary",
        )

        # 3. Either fit or directly predict
        preds = trainer_new.predict(X_wide=X_wide, X_tab=X_tab, batch_size=256)
