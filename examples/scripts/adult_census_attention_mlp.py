import numpy as np
import torch
import pandas as pd

from pytorch_widedeep import Trainer
from pytorch_widedeep.models import (
    WideDeep,
    SelfAttentionMLP,
    ContextAttentionMLP,
)
from pytorch_widedeep.metrics import Accuracy
from pytorch_widedeep.datasets import load_adult
from pytorch_widedeep.preprocessing import TabPreprocessor

use_cuda = torch.cuda.is_available()

if __name__ == "__main__":
    df = load_adult(as_frame=True)
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

    tab_preprocessor = TabPreprocessor(
        cat_embed_cols=cat_embed_cols,
        continuous_cols=continuous_cols,
        with_attention=True,
        scale=True,
    )
    X_tab = tab_preprocessor.fit_transform(df)

    for attention_name in ["context_attention", "self_attention"]:
        if attention_name == "context_attention":
            tab_mlp_attn = ContextAttentionMLP(
                column_idx=tab_preprocessor.column_idx,
                cat_embed_input=tab_preprocessor.cat_embed_input,
                continuous_cols=continuous_cols,
                input_dim=16,
                attn_dropout=0.2,
                n_blocks=3,
            )
        elif attention_name == "self_attention":
            tab_mlp_attn = SelfAttentionMLP(  # type: ignore[assignment]
                column_idx=tab_preprocessor.column_idx,
                cat_embed_input=tab_preprocessor.cat_embed_input,
                continuous_cols=continuous_cols,
                input_dim=16,
                attn_dropout=0.2,
                n_blocks=3,
            )

        model = WideDeep(deeptabular=tab_mlp_attn)
        # tab_opt = torch.optim.AdamW(model.deeptabular.parameters(), lr=0.01)

        trainer = Trainer(
            model,
            # optimizers=tab_opt,
            objective="binary",
            metrics=[Accuracy],
        )
        trainer.fit(
            X_tab=X_tab,
            target=target,
            n_epochs=2,
            batch_size=256,
            val_split=0.2,
        )
