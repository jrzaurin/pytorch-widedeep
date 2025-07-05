import os
import shutil

import numpy as np
import torch
import pandas as pd
import pytest

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


@pytest.mark.parametrize("save_state_dict", [True, False])
def test_save_one_optimizer(save_state_dict):

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
        save_state_dict=save_state_dict,
        save_optimizer=True,
        model_filename="model_and_optimizer.pt",
    )

    checkpoint = torch.load(
        os.path.join(save_path, "model_and_optimizer.pt"), weights_only=False
    )

    if save_state_dict:
        new_model = WideDeep(wide=wide, deeptabular=tab_mlp)
        # just to change the initial weights
        new_model.wide.wide_linear.weight.data = torch.nn.init.xavier_normal_(
            new_model.wide.wide_linear.weight
        )
        new_optimizer = torch.optim.AdamW(new_model.parameters(), lr=0.001)
        new_model.load_state_dict(checkpoint["model_state_dict"])
        new_optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    else:
        # This else statement is mostly testing that it runs, as it does not
        # involved loading a state_dict
        saved_objects = torch.load(
            os.path.join(save_path, "model_and_optimizer.pt"), weights_only=False
        )
        new_model = saved_objects["model"]
        new_optimizer = saved_objects["optimizer"]

    shutil.rmtree(save_path)

    assert torch.all(
        new_model.wide.wide_linear.weight.data == model.wide.wide_linear.weight.data
    ) and torch.all(
        new_optimizer.state_dict()["state"][1]["exp_avg"]
        == trainer.optimizer.state_dict()["state"][1]["exp_avg"]
    )


@pytest.mark.parametrize("save_state_dict", [True, False])
def test_save_multiple_optimizers(save_state_dict):

    model = WideDeep(wide=wide, deeptabular=tab_mlp)

    wide_opt = torch.optim.AdamW(model.wide.parameters(), lr=0.001)
    deep_opt = torch.optim.AdamW(model.deeptabular.parameters(), lr=0.001)

    optimizers = {"wide": wide_opt, "deeptabular": deep_opt}

    trainer = Trainer(
        model,
        objective="binary",
        optimizers=optimizers,
        metrics=[Accuracy()],
    )

    trainer.fit(X_wide=X_wide, X_tab=X_tab, target=df["target"].values, n_epochs=1)

    trainer.save(
        path=save_path,
        save_state_dict=save_state_dict,
        save_optimizer=True,
        model_filename="model_and_optimizer.pt",
    )

    checkpoint = torch.load(
        os.path.join(save_path, "model_and_optimizer.pt"), weights_only=False
    )

    if save_state_dict:
        new_model = WideDeep(wide=wide, deeptabular=tab_mlp)
        # just to change the initial weights
        new_model.wide.wide_linear.weight.data = torch.nn.init.xavier_normal_(
            new_model.wide.wide_linear.weight
        )

        new_wide_opt = torch.optim.AdamW(model.wide.parameters(), lr=0.001)
        new_deep_opt = torch.optim.AdamW(model.deeptabular.parameters(), lr=0.001)
        new_model.load_state_dict(checkpoint["model_state_dict"])
        new_wide_opt.load_state_dict(checkpoint["optimizer_state_dict"]["wide"])
        new_deep_opt.load_state_dict(checkpoint["optimizer_state_dict"]["deeptabular"])
    else:
        # This else statement is mostly testing that it runs, as it does not
        # involved loading a state_dict
        saved_objects = torch.load(
            os.path.join(save_path, "model_and_optimizer.pt"), weights_only=False
        )
        new_model = saved_objects["model"]
        new_optimizers = saved_objects["optimizer"]
        new_wide_opt = new_optimizers._optimizers["wide"]
        new_deep_opt = new_optimizers._optimizers["deeptabular"]

    shutil.rmtree(save_path)

    assert (
        torch.all(
            new_model.wide.wide_linear.weight.data == model.wide.wide_linear.weight.data
        )
        and torch.all(
            new_wide_opt.state_dict()["state"][1]["exp_avg"]
            == trainer.optimizer._optimizers["wide"].state_dict()["state"][1]["exp_avg"]
        )
        and torch.all(
            new_deep_opt.state_dict()["state"][1]["exp_avg"]
            == trainer.optimizer._optimizers["deeptabular"].state_dict()["state"][1][
                "exp_avg"
            ]
        )
    )
