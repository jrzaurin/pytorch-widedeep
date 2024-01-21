import os
import shutil
import string

import numpy as np
import torch
import pandas as pd
import pytest
from torch.optim.lr_scheduler import StepLR

from pytorch_widedeep.models import TabMlp, TabTransformer
from pytorch_widedeep.callbacks import LRHistory
from pytorch_widedeep.preprocessing import TabPreprocessor
from pytorch_widedeep.self_supervised_training import (
    EncoderDecoderTrainer,
    ContrastiveDenoisingTrainer,
)

some_letters = list(string.ascii_lowercase)
some_numbers = range(10)
df = pd.DataFrame(
    {
        "col1": list(np.random.choice(some_letters, 32)),
        "col2": list(np.random.choice(some_letters, 32)),
        "col3": list(np.random.choice(some_numbers, 32)),
        "col4": list(np.random.choice(some_numbers, 32)),
    }
)

non_transf_preprocessor = TabPreprocessor(
    cat_embed_cols=["col1", "col2"],
    continuous_cols=["col3", "col4"],
)
X_tab = non_transf_preprocessor.fit_transform(df)


transf_preprocessor = TabPreprocessor(
    cat_embed_cols=["col1", "col2"],
    continuous_cols=["col3", "col4"],
    with_attention=True,
)
X_tab_transf = transf_preprocessor.fit_transform(df)


###############################################################################
# test save
###############################################################################


@pytest.mark.parametrize(
    "model_type",
    ["mlp", "transformer"],
)
def test_save_and_load(model_type):
    if model_type == "mlp":
        model = TabMlp(
            column_idx=non_transf_preprocessor.column_idx,
            cat_embed_input=non_transf_preprocessor.cat_embed_input,
            continuous_cols=non_transf_preprocessor.continuous_cols,
            mlp_hidden_dims=[16, 8],
        )
        X = X_tab
    elif model_type == "transformer":
        model = TabTransformer(
            column_idx=transf_preprocessor.column_idx,
            cat_embed_input=transf_preprocessor.cat_embed_input,
            continuous_cols=transf_preprocessor.continuous_cols,
            embed_continuous=True,
            embed_continuous_method="standard",
            n_heads=2,
            n_blocks=2,
        )
        X = X_tab_transf

    optimizer = torch.optim.Adam(model.parameters())
    lr_scheduler = StepLR(optimizer, step_size=4)

    if model_type == "mlp":
        trainer = EncoderDecoderTrainer(
            encoder=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            callbacks=[LRHistory(n_epochs=5)],
            masked_prob=0.2,
            verbose=0,
        )
    elif model_type == "transformer":
        trainer = ContrastiveDenoisingTrainer(
            model=model,
            preprocessor=transf_preprocessor,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            callbacks=[LRHistory(n_epochs=5)],
            verbose=0,
        )

    trainer.pretrain(X, n_epochs=5, batch_size=16)

    if model_type == "mlp":
        col_embed_module = model.cat_embed.embed_layers.emb_layer_col1
        embeddings = col_embed_module.weight.data
    elif model_type == "transformer":
        embed_module = model.cat_embed.embed
        embeddings = embed_module.weight.data

    trainer.save("tests/test_self_supervised/model_dir/", model_filename="ss_model.pt")
    new_model = torch.load("tests/test_self_supervised/model_dir/ss_model.pt")

    if model_type == "mlp":
        new_col_embed_module = new_model.encoder.cat_embed.embed_layers.emb_layer_col1
        new_embeddings = new_col_embed_module.weight.data
    elif model_type == "transformer":
        new_embed_module = new_model.model.cat_embed.embed
        new_embeddings = new_embed_module.weight.data

    shutil.rmtree("tests/test_self_supervised/model_dir/")
    assert torch.allclose(embeddings, new_embeddings)


def _build_model_and_trainer(model_type):
    if model_type == "mlp":
        model = TabMlp(
            column_idx=non_transf_preprocessor.column_idx,
            cat_embed_input=non_transf_preprocessor.cat_embed_input,
            continuous_cols=non_transf_preprocessor.continuous_cols,
            mlp_hidden_dims=[16, 8],
        )
        trainer = EncoderDecoderTrainer(
            encoder=model,
            masked_prob=0.2,
            verbose=0,
        )
    elif model_type == "transformer":
        model = TabTransformer(
            column_idx=transf_preprocessor.column_idx,
            cat_embed_input=transf_preprocessor.cat_embed_input,
            continuous_cols=transf_preprocessor.continuous_cols,
            embed_continuous=True,
            embed_continuous_method="standard",
            n_heads=2,
            n_blocks=2,
        )
        trainer = ContrastiveDenoisingTrainer(
            model=model,
            preprocessor=transf_preprocessor,
            verbose=0,
        )

    return model, trainer


@pytest.mark.parametrize(
    "model_type",
    ["mlp", "transformer"],
)
def test_save_and_load_dict(model_type):  # noqa: C901
    model1, trainer1 = _build_model_and_trainer(model_type)
    X = X_tab if model_type == "mlp" else X_tab_transf

    trainer1.pretrain(X, n_epochs=5, batch_size=16)

    if model_type == "mlp":
        col_embed_module = model1.cat_embed.embed_layers.emb_layer_col1
        embeddings = col_embed_module.weight.data
    elif model_type == "transformer":
        embed_module = model1.cat_embed.embed
        embeddings = embed_module.weight.data

    trainer1.save(
        "tests/test_self_supervised/model_dir/",
        model_filename="ss_model.pt",
        save_state_dict=True,
    )

    model2, trainer2 = _build_model_and_trainer(model_type)

    if model_type == "mlp":
        trainer2.ed_model.load_state_dict(
            torch.load("tests/test_self_supervised/model_dir/ss_model.pt")
        )
    elif model_type == "transformer":
        trainer2.cd_model.load_state_dict(
            torch.load("tests/test_self_supervised/model_dir/ss_model.pt")
        )

    if model_type == "mlp":
        new_col_embed_module = (
            trainer2.ed_model.encoder.cat_embed.embed_layers.emb_layer_col1
        )
        new_embeddings = new_col_embed_module.weight.data
    elif model_type == "transformer":
        new_embed_module = trainer2.cd_model.model.cat_embed.embed
        new_embeddings = new_embed_module.weight.data

    same_weights = torch.allclose(embeddings, new_embeddings)

    if os.path.isfile(
        "tests/test_self_supervised/model_dir/history/train_eval_history.json"
    ):
        history_saved = True
    else:
        history_saved = False
    shutil.rmtree("tests/test_self_supervised/model_dir/")

    assert same_weights and history_saved
