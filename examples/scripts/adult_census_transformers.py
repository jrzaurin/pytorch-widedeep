import numpy as np
import torch
import pandas as pd

from pytorch_widedeep import Trainer
from pytorch_widedeep.models import (
    SAINT,
    Wide,
    WideDeep,
    TabPerceiver,
    FTTransformer,
    TabFastFormer,
    TabTransformer,
)
from pytorch_widedeep.metrics import Accuracy
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
    df.head()

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
        "education",
        "relationship",
        "workclass",
        "occupation",
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
        scale=True,
        for_transformer=True,
        with_cls_token=True,
    )
    X_tab = tab_preprocessor.fit_transform(df)

    wide = Wide(input_dim=np.unique(X_wide).shape[0], pred_dim=1)

    tab_transformer = TabTransformer(
        column_idx=tab_preprocessor.column_idx,
        cat_embed_input=tab_preprocessor.cat_embed_input,
        continuous_cols=continuous_cols,
        embed_continuous_method="standard",
        n_blocks=4,
    )

    saint = SAINT(
        column_idx=tab_preprocessor.column_idx,
        cat_embed_input=tab_preprocessor.cat_embed_input,
        continuous_cols=continuous_cols,
        cont_norm_layer="batchnorm",
        n_blocks=4,
    )

    tab_perceiver = TabPerceiver(
        column_idx=tab_preprocessor.column_idx,
        cat_embed_input=tab_preprocessor.cat_embed_input,
        continuous_cols=continuous_cols,
        n_latents=6,
        latent_dim=16,
        n_latent_blocks=4,
        n_perceiver_blocks=2,
        share_weights=False,
    )

    tab_fastformer = TabFastFormer(
        column_idx=tab_preprocessor.column_idx,
        cat_embed_input=tab_preprocessor.cat_embed_input,
        continuous_cols=continuous_cols,
        n_blocks=4,
        n_heads=4,
        share_qv_weights=False,
        share_weights=False,
    )

    ft_transformer = FTTransformer(
        column_idx=tab_preprocessor.column_idx,
        cat_embed_input=tab_preprocessor.cat_embed_input,
        continuous_cols=continuous_cols,
        input_dim=32,
        kv_compression_factor=0.5,
        n_blocks=3,
        n_heads=4,
    )

    for tab_model in [
        tab_transformer,
        saint,
        ft_transformer,
        tab_perceiver,
        tab_fastformer,
    ]:
        model = WideDeep(wide=wide, deeptabular=tab_model)

        wide_opt = torch.optim.Adam(model.wide.parameters(), lr=0.01)
        deep_opt = torch.optim.Adam(model.wide.parameters(), lr=0.01)
        wide_sch = torch.optim.lr_scheduler.StepLR(wide_opt, step_size=3)
        deep_sch = torch.optim.lr_scheduler.StepLR(deep_opt, step_size=5)

        optimizers = {"wide": wide_opt, "deeptabular": deep_opt}
        schedulers = {"wide": wide_sch, "deeptabular": deep_sch}
        initializers = {"wide": KaimingNormal, "deeptabular": XavierNormal}
        callbacks = [
            LRHistory(n_epochs=10),
            EarlyStopping(patience=5),
            ModelCheckpoint(filepath="model_weights/wd_out"),
        ]
        metrics = [Accuracy]

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
            n_epochs=1,
            batch_size=128,
            val_split=0.2,
        )
