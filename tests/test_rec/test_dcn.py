import torch

from pytorch_widedeep import Trainer
from pytorch_widedeep.models import Wide, WideDeep
from pytorch_widedeep.models.rec import DeepCrossNetwork
from pytorch_widedeep.preprocessing import TabPreprocessor, WidePreprocessor

from .utils_test_rec import create_train_val_test_data

train, valid, test = create_train_val_test_data()


def test_dcn_forward(cat_embed_cols, continuous_cols):
    tab_preprocessor = TabPreprocessor(
        cat_embed_cols=cat_embed_cols,
        continuous_cols=continuous_cols,
    )

    X_tab = tab_preprocessor.fit_transform(train)
    X_tab_tnsr = torch.tensor(X_tab, dtype=torch.float32)

    model = DeepCrossNetwork(
        column_idx=tab_preprocessor.column_idx,
        cat_embed_input=tab_preprocessor.cat_embed_input,  # type: ignore[arg-type]
        continuous_cols=continuous_cols,
        n_cross_layers=2,
        mlp_hidden_dims=[16, 8],
    )

    res = model(X_tab_tnsr)

    assert res.shape == (X_tab_tnsr.shape[0], model.output_dim)


def test_dcn_with_wide_component(cat_embed_cols):
    tab_preprocessor = TabPreprocessor(
        cat_embed_cols=cat_embed_cols,
    )

    X_tab = tab_preprocessor.fit_transform(train)
    X_tab_tnsr = torch.tensor(X_tab, dtype=torch.float32)

    wide_preprocessor = WidePreprocessor(wide_cols=cat_embed_cols)
    X_wide = wide_preprocessor.fit_transform(train)
    X_wide_tnsr = torch.tensor(X_wide, dtype=torch.float32)

    dcn = DeepCrossNetwork(
        column_idx=tab_preprocessor.column_idx,
        cat_embed_input=tab_preprocessor.cat_embed_input,  # type: ignore[arg-type]
        n_cross_layers=2,
        mlp_hidden_dims=[16, 8],
    )

    linear = Wide(input_dim=X_wide.max())

    model = WideDeep(wide=linear, deeptabular=dcn)

    X_inp = {"wide": X_wide_tnsr, "deeptabular": X_tab_tnsr}

    out = model(X_inp)

    assert out.shape[0] == X_tab_tnsr.shape[0]
    assert out.shape[1] == 1


def test_dcn_full_process(cat_embed_cols):
    tab_preprocessor = TabPreprocessor(
        cat_embed_cols=cat_embed_cols,
    )

    X_tab_tr = tab_preprocessor.fit_transform(train)
    X_tab_val = tab_preprocessor.transform(valid)
    X_tab_te = tab_preprocessor.transform(test)
    y_tr = train["purchased"].values
    y_val = valid["purchased"].values

    dcn = DeepCrossNetwork(
        column_idx=tab_preprocessor.column_idx,
        cat_embed_input=tab_preprocessor.cat_embed_input,  # type: ignore[arg-type]
        n_cross_layers=2,
        mlp_hidden_dims=[16, 8],
    )
    model = WideDeep(deeptabular=dcn)

    X_train = {"X_tab": X_tab_tr, "target": y_tr}
    X_val = {"X_tab": X_tab_val, "target": y_val}
    trainer = Trainer(model=model, objective="binary", verbose=0)
    trainer.fit(X_train=X_train, X_val=X_val, n_epochs=1)
    preds = trainer.predict(X_tab=X_tab_te)

    assert preds.shape[0] == X_tab_te.shape[0]
    assert (
        trainer.history is not None
        and "train_loss" in trainer.history
        and "val_loss" in trainer.history
    )
