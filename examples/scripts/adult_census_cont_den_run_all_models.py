from itertools import product

import numpy as np
import torch
import pandas as pd

from pytorch_widedeep.models import (
    SAINT,
    FTTransformer,
    TabFastFormer,
    TabTransformer,
    SelfAttentionMLP,
    ContextAttentionMLP,
)
from pytorch_widedeep.datasets import load_adult
from pytorch_widedeep.preprocessing import TabPreprocessor
from pytorch_widedeep.self_supervised_training import ContrastiveDenoisingTrainer

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

    transformer_models = [
        "tab_transformer",
        "saint",
        "tab_fastformer",
        "ft_transformer",
    ]
    with_cls_token = [True, False]

    for w_cls_tok, transf_model in product(with_cls_token, transformer_models):
        processor = TabPreprocessor(
            cat_embed_cols=cat_embed_cols,
            continuous_cols=continuous_cols,
            scale=True,
            with_attention=True,
            with_cls_token=w_cls_tok,
        )
        X_tab = processor.fit_transform(df)

        if transf_model == "tab_transformer":
            model = TabTransformer(
                column_idx=processor.column_idx,
                cat_embed_input=processor.cat_embed_input,
                continuous_cols=continuous_cols,
                embed_continuous=True,
                embed_continuous_method="standard",
                n_blocks=4,
            )
        if transf_model == "saint":
            model = SAINT(  # type: ignore[assignment]
                column_idx=processor.column_idx,
                cat_embed_input=processor.cat_embed_input,
                continuous_cols=continuous_cols,
                cont_norm_layer="batchnorm",
                n_blocks=4,
            )
        if transf_model == "tab_fastformer":
            model = TabFastFormer(  # type: ignore[assignment]
                column_idx=processor.column_idx,
                cat_embed_input=processor.cat_embed_input,
                continuous_cols=continuous_cols,
                n_blocks=4,
                n_heads=4,
                share_qv_weights=False,
                share_weights=False,
            )
        if transf_model == "ft_transformer":
            model = FTTransformer(  # type: ignore[assignment]
                column_idx=processor.column_idx,
                cat_embed_input=processor.cat_embed_input,
                continuous_cols=continuous_cols,
                input_dim=32,
                kv_compression_factor=0.5,
                n_blocks=3,
                n_heads=4,
            )

        ss_trainer = ContrastiveDenoisingTrainer(
            model=model,
            preprocessor=processor,
        )
        ss_trainer.pretrain(X_tab, n_epochs=1, batch_size=256)

    mlp_attn_model = ["context_attention", "self_attention"]

    for w_cls_tok, attn_model in product(with_cls_token, mlp_attn_model):
        processor = TabPreprocessor(
            cat_embed_cols=cat_embed_cols,
            continuous_cols=continuous_cols,
            with_attention=True,
            with_cls_token=w_cls_tok,
        )
        X_tab = processor.fit_transform(df)

        if attn_model == "context_attention":
            model = ContextAttentionMLP(  # type: ignore[assignment]
                column_idx=processor.column_idx,
                cat_embed_input=processor.cat_embed_input,
                continuous_cols=continuous_cols,
                input_dim=16,
                attn_dropout=0.2,
                n_blocks=3,
            )
        if attn_model == "self_attention":
            model = SelfAttentionMLP(  # type: ignore[assignment]
                column_idx=processor.column_idx,
                cat_embed_input=processor.cat_embed_input,
                continuous_cols=continuous_cols,
                input_dim=16,
                attn_dropout=0.2,
                n_blocks=3,
            )

        ss_trainer = ContrastiveDenoisingTrainer(
            model=model,
            preprocessor=processor,
        )
        ss_trainer.pretrain(X_tab, n_epochs=1, batch_size=256)
