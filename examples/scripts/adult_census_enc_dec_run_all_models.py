import numpy as np
import torch
import pandas as pd

from pytorch_widedeep.models import TabMlp as TabMlpEncoder
from pytorch_widedeep.models import TabNet as TabNetEncoder
from pytorch_widedeep.models import TabResnet as TabResnetEncoder
from pytorch_widedeep.models import TabMlpDecoder, TabNetDecoder, TabResnetDecoder
from pytorch_widedeep.datasets import load_adult
from pytorch_widedeep.preprocessing import TabPreprocessor
from pytorch_widedeep.self_supervised_training import EncoderDecoderTrainer

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
        scale=True,
    )
    X_tab = tab_preprocessor.fit_transform(df)

    mlp_encoder = TabMlpEncoder(
        column_idx=tab_preprocessor.column_idx,
        cat_embed_input=tab_preprocessor.cat_embed_input,
        continuous_cols=continuous_cols,
        mlp_hidden_dims=[200, 100],
    )

    mlp_decoder = TabMlpDecoder(
        embed_dim=95,
        mlp_hidden_dims=[100, 200],
    )

    resnet_encoder = TabResnetEncoder(
        column_idx=tab_preprocessor.column_idx,
        cat_embed_input=tab_preprocessor.cat_embed_input,
        continuous_cols=continuous_cols,
        blocks_dims=[200, 100, 100],
    )

    resnet_decoder = TabResnetDecoder(
        embed_dim=95,
        blocks_dims=[100, 100, 200],
    )

    tabnet_encoder = TabNetEncoder(
        column_idx=tab_preprocessor.column_idx,
        cat_embed_input=tab_preprocessor.cat_embed_input,
        continuous_cols=continuous_cols,
    )

    tabnet_decoder = TabNetDecoder(
        embed_dim=95,
    )

    encoders = [mlp_encoder, resnet_encoder, tabnet_encoder] + [
        mlp_encoder,
        resnet_encoder,
        tabnet_encoder,
    ]
    decoders = [mlp_decoder, resnet_decoder, tabnet_decoder, None, None, None]

    for enc, dec in zip(encoders, decoders):
        ec_trainer = EncoderDecoderTrainer(
            encoder=enc,  # type: ignore[arg-type]
            decoder=dec,  # type: ignore[arg-type]
            masked_prob=0.2,
        )
        ec_trainer.pretrain(X_tab, n_epochs=1, batch_size=256)
