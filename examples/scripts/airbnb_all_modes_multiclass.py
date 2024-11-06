from pathlib import Path

import numpy as np
import torch
import pandas as pd

import pytorch_widedeep as wd
from pytorch_widedeep.models import Wide, TabMlp, WideDeep
from pytorch_widedeep.metrics import F1Score, Precision
from pytorch_widedeep.preprocessing import TabPreprocessor, WidePreprocessor

if __name__ == "__main__":
    # The airbnb dataset, which you could get from here:
    # http://insideairbnb.com/get-the-data.html, is too big to be included in
    # our datasets module (when including images). Therefore, go there,
    # download it, and use the download_images.py script to get the images
    # and the airbnb_data_processing.py to process the data. We'll find
    # better datasets in the future ;). Note that here we are only using a
    # small sample to illustrate the use, so PLEASE ignore the results, just
    # focus on usage
    DATA_PATH = Path("../tmp_data")

    df = pd.read_csv(DATA_PATH / "airbnb/airbnb_sample.csv")

    crossed_cols = [("property_type", "room_type")]
    already_dummies = [c for c in df.columns if "amenity" in c] + ["has_house_rules"]
    wide_cols = [
        "is_location_exact",
        "property_type",
        "room_type",
        "host_gender",
        "instant_bookable",
    ] + already_dummies
    already_standard = ["latitude", "longitude"]

    cat_embed_cols = [(c, 16) for c in df.columns if "catg" in c] + [
        ("neighbourhood_cleansed", 64),
        ("cancellation_policy", 16),
    ]
    continuous_cols = ["latitude", "longitude", "security_deposit", "extra_people"]

    df["yield_cat"] = pd.cut(df["yield"], bins=[0.2, 65, 163, 600], labels=[0, 1, 2])
    df.drop("yield", axis=1, inplace=True)
    target = "yield_cat"
    target = np.array(df[target].values)  # type: ignore[assignment]

    wide_preprocessor = WidePreprocessor(wide_cols=wide_cols, crossed_cols=crossed_cols)
    X_wide = wide_preprocessor.fit_transform(df)

    tab_preprocessor = TabPreprocessor(
        cat_embed_cols=cat_embed_cols, continuous_cols=continuous_cols, scale=True  # type: ignore[arg-type]
    )
    X_tab = tab_preprocessor.fit_transform(df)

    wide = Wide(input_dim=np.unique(X_wide).shape[0], pred_dim=3)
    deepdense = TabMlp(
        mlp_hidden_dims=[64, 32],
        mlp_dropout=[0.2, 0.2],
        column_idx=tab_preprocessor.column_idx,
        cat_embed_input=tab_preprocessor.cat_embed_input,
        continuous_cols=continuous_cols,
    )
    model = WideDeep(wide=wide, deeptabular=deepdense, pred_dim=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03)

    trainer = wd.Trainer(
        model,
        objective="multiclass",
        metrics=[Precision(average=False), F1Score],
        optimizers=optimizer,
    )

    trainer.fit(
        X_wide=X_wide,
        X_tab=X_tab,
        target=target,
        n_epochs=1,
        batch_size=32,
        val_split=0.2,
    )
