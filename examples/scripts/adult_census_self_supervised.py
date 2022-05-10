import numpy as np
import torch
import pandas as pd

from pytorch_widedeep.models import (
    SAINT,
    TabPerceiver,
    FTTransformer,
    TabFastFormer,
    TabTransformer,
)
from pytorch_widedeep.datasets import load_adult
from pytorch_widedeep.preprocessing import TabPreprocessor
from pytorch_widedeep.self_supervised_training.self_supervised_trainer import (
    SelfSupervisedTrainer,
)

use_cuda = torch.cuda.is_available()

if __name__ == "__main__":

    df = load_adult(as_frame=True)
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
    target = "income_label"
    target = df[target].values

    tab_preprocessor = TabPreprocessor(
        cat_embed_cols=cat_embed_cols,
        continuous_cols=continuous_cols,
        with_attention=True,
        with_cls_token=True,
    )
    X_tab = tab_preprocessor.fit_transform(df)

    tab_transformer = TabTransformer(
        column_idx=tab_preprocessor.column_idx,
        cat_embed_input=tab_preprocessor.cat_embed_input,
        continuous_cols=continuous_cols,
        embed_continuous=True,
        n_blocks=4,
    )

    saint = SAINT(
        column_idx=tab_preprocessor.column_idx,
        cat_embed_input=tab_preprocessor.cat_embed_input,
        continuous_cols=continuous_cols,
        cont_norm_layer="batchnorm",
        n_blocks=4,
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

    for transformer_model in [tab_transformer, saint, tab_fastformer, ft_transformer]:
        ss_trainer = SelfSupervisedTrainer(
            model=transformer_model,
            preprocessor=tab_preprocessor,
        )
        ss_trainer.pretrain(X_tab, n_epochs=1, batch_size=256)
