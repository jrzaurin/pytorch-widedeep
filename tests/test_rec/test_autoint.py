import torch
import pytest

from pytorch_widedeep import Trainer
from pytorch_widedeep.models import Wide, WideDeep
from pytorch_widedeep.models.rec import AutoInt
from pytorch_widedeep.preprocessing import TabPreprocessor, WidePreprocessor

from .utils_test_rec import create_train_val_test_data

train, valid, test = create_train_val_test_data()


@pytest.mark.parametrize("reduction", ["mean", "cat"])
def test_autoint_reduction(reduction, cat_embed_cols, continuous_cols):
    tab_preprocessor = TabPreprocessor(
        cat_embed_cols=cat_embed_cols,
        continuous_cols=continuous_cols,
        with_attention=True,
    )

    X_tab = tab_preprocessor.fit_transform(train)
    X_tab_tnsr = torch.tensor(X_tab, dtype=torch.float32)

    model = AutoInt(
        column_idx=tab_preprocessor.column_idx,
        input_dim=16,
        cat_embed_input=tab_preprocessor.cat_embed_input,  # type: ignore[arg-type]
        num_heads=2,
        num_layers=2,
        reduction=reduction,
        continuous_cols=continuous_cols,
        embed_continuous_method="standard",
    )

    res = model(X_tab_tnsr)

    assert res.shape == (X_tab_tnsr.shape[0], model.output_dim)


def test_autoint_with_wide_component(cat_embed_cols):
    tab_preprocessor = TabPreprocessor(
        cat_embed_cols=cat_embed_cols,
        with_attention=True,
    )

    X_tab = tab_preprocessor.fit_transform(train)
    X_tab_tnsr = torch.tensor(X_tab, dtype=torch.float32)

    wide_preprocessor = WidePreprocessor(wide_cols=cat_embed_cols)
    X_wide = wide_preprocessor.fit_transform(train)
    X_wide_tnsr = torch.tensor(X_wide, dtype=torch.float32)

    auto_int = AutoInt(
        column_idx=tab_preprocessor.column_idx,
        input_dim=16,
        cat_embed_input=tab_preprocessor.cat_embed_input,  # type: ignore[arg-type]
        num_heads=2,
    )

    linear = Wide(input_dim=X_wide.max())

    model = WideDeep(wide=linear, deeptabular=auto_int)

    X_inp = {"wide": X_wide_tnsr, "deeptabular": X_tab_tnsr}

    out = model(X_inp)

    assert out.shape[0] == X_tab_tnsr.shape[0]
    assert out.shape[1] == 1


def test_autoint_full_process(cat_embed_cols):

    tab_preprocessor = TabPreprocessor(
        cat_embed_cols=cat_embed_cols,
        for_mf=True,
    )

    X_tab_tr = tab_preprocessor.fit_transform(train)
    X_tab_val = tab_preprocessor.transform(valid)
    X_tab_te = tab_preprocessor.transform(test)
    y_tr = train["purchased"].values
    y_val = valid["purchased"].values

    auto_int = AutoInt(
        column_idx=tab_preprocessor.column_idx,
        input_dim=16,
        cat_embed_input=tab_preprocessor.cat_embed_input,  # type: ignore[arg-type]
        num_heads=2,
    )
    model = WideDeep(deeptabular=auto_int)

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


@pytest.mark.parametrize(
    "embed_continuous_method", ["periodic", "piecewise", "standard"]
)
def test_autoint_cont_embed_methods(
    embed_continuous_method, cat_embed_cols, continuous_cols
):
    tab_preprocessor = TabPreprocessor(
        cat_embed_cols=cat_embed_cols,
        continuous_cols=continuous_cols,
        with_attention=True,
    )

    X_tab = tab_preprocessor.fit_transform(train)
    X_tab_tnsr = torch.tensor(X_tab, dtype=torch.float32)

    if embed_continuous_method == "periodic":
        model = AutoInt(
            column_idx=tab_preprocessor.column_idx,
            input_dim=16,
            cat_embed_input=tab_preprocessor.cat_embed_input,
            num_heads=2,
            num_layers=2,
            continuous_cols=continuous_cols,
            embed_continuous_method=embed_continuous_method,
            n_frequencies=4,
            sigma=0.1,
            share_last_layer=False,
        )
    else:
        quantization_setup = {
            "item_price": list(train["item_price"].quantile([0.25, 0.5, 0.75]).values),
            "user_age": list(train["user_age"].quantile([0.25, 0.5, 0.75]).values),
        }
        model = AutoInt(
            column_idx=tab_preprocessor.column_idx,
            input_dim=16,
            cat_embed_input=tab_preprocessor.cat_embed_input,
            num_heads=2,
            num_layers=2,
            continuous_cols=continuous_cols,
            embed_continuous_method=embed_continuous_method,
            quantization_setup=quantization_setup,
        )

    res = model(X_tab_tnsr)

    assert res.shape == (X_tab_tnsr.shape[0], model.output_dim)
