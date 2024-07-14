import os
import shutil

import numpy as np
import torch
import pandas as pd

from pytorch_widedeep import Trainer
from pytorch_widedeep.models import Wide, TabMlp, WideDeep
from pytorch_widedeep.metrics import Accuracy
from pytorch_widedeep.preprocessing import TabPreprocessor, WidePreprocessor

full_path = os.path.realpath(__file__)
path = os.path.split(full_path)[0]
save_path = os.path.join(path, "test_save_optimizer_dir")


data = {
    "categorical_1": ["a", "b", "c", "d"] * 16,
    "categorical_2": ["e", "f", "g", "h"] * 16,
    "continuous_1": [1, 2, 3, 4] * 16,
    "continuous_2": [5, 6, 7, 8] * 16,
    "target": [0, 1] * 32,
}

df = pd.DataFrame(data)


cat_cols = ["categorical_1", "categorical_2"]
wide_preprocessor = WidePreprocessor(wide_cols=cat_cols)
X_wide = wide_preprocessor.fit_transform(df)

tab_preprocessor = TabPreprocessor(
    cat_embed_cols=cat_cols,
    continuous_cols=["continuous_1", "continuous_2"],
    scale=True,
)
X_tab = tab_preprocessor.fit_transform(df)

wide = Wide(input_dim=np.unique(X_wide).shape[0], pred_dim=1)

tab_mlp = TabMlp(
    column_idx=tab_preprocessor.column_idx,
    cat_embed_input=tab_preprocessor.cat_embed_input,
    continuous_cols=["continuous_1", "continuous_2"],
    mlp_hidden_dims=[16, 8],
)

# wide_opt = torch.optim.AdamW(model.wide.parameters(), lr=0.001)
# deep_opt = torch.optim.AdamW(model.deeptabular.parameters(), lr=0.001)

# optimizers = {"wide": wide_opt, "deeptabular": deep_opt}


def test_save_one_optimizer():

    model = WideDeep(wide=wide, deeptabular=tab_mlp)

    trainer = Trainer(
        model,
        objective="binary",
        optimizer=torch.optim.AdamW(model.parameters(), lr=0.001),
        metrics=[Accuracy()],
    )

    trainer.fit(X_wide=X_wide, X_tab=X_tab, target=df["target"].values, n_epochs=1)

    trainer.save(
        path=save_path,
        save_state_dict=True,
        save_optimizer=True,
        model_filename="model_and_optimizer.pt",
    )

    checkpoint = torch.load(os.path.join(save_path, "model_and_optimizer.pt"))

    new_model = WideDeep(wide=wide, deeptabular=tab_mlp)
    # just to change the initial weights
    new_model.wide.wide_linear.weight.data = torch.nn.init.xavier_normal_(
        new_model.wide.wide_linear.weight
    )
    new_optimizer = torch.optim.AdamW(new_model.parameters(), lr=0.001)

    new_model.load_state_dict(checkpoint["model_state_dict"])
    new_optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    shutil.rmtree(save_path)

    assert torch.all(
        new_model.wide.wide_linear.weight.data == model.wide.wide_linear.weight.data
    ) and torch.all(
        new_optimizer.state_dict()["state"][1]["exp_avg"]
        == trainer.optimizer.state_dict()["state"][1]["exp_avg"]
    )
