import torch
import pytest

from pytorch_widedeep import Trainer
from pytorch_widedeep.models import Wide, WideDeep
from pytorch_widedeep.models.rec import DeepFactorizationMachine
from pytorch_widedeep.preprocessing import TabPreprocessor, WidePreprocessor

from .utils_test_rec import create_train_val_test_data

train, valid, test = create_train_val_test_data()


@pytest.mark.parametrize(
    "reduce_sum, mlp_hidden_dims",
    [
        (True, None),
        (False, None),
        (True, [16, 8]),
        (False, [16, 8]),
    ],
)
def test_deepfm_reduce_sum(
    reduce_sum, mlp_hidden_dims, cat_embed_cols, continuous_cols
):

    tab_preprocessor = TabPreprocessor(
        cat_embed_cols=cat_embed_cols,
        continuous_cols=continuous_cols,
        for_mf=True,
    )

    X_tab = tab_preprocessor.fit_transform(train)

    model = DeepFactorizationMachine(
        column_idx=tab_preprocessor.column_idx,
        num_factors=8,
        reduce_sum=reduce_sum,
        cat_embed_input=tab_preprocessor.cat_embed_input,
        continuous_cols=continuous_cols,
        embed_continuous_method="periodic",
        n_frequencies=5,
        sigma=0.1,
        share_last_layer=False,
        mlp_hidden_dims=mlp_hidden_dims,
    )

    X_tab_tnsr = torch.tensor(X_tab)
    res = model(X_tab_tnsr)

    if reduce_sum:
        assert res.shape == (X_tab_tnsr.shape[0], 1)
    else:
        assert res.shape == (X_tab_tnsr.shape[0], model.output_dim)


@pytest.mark.parametrize(
    "embed_continuous_method", ["periodic", "piecewise", "standard"]
)
@pytest.mark.parametrize("reduce_sum", [True, False])
def test_deepfm_cont_embed_methods(
    embed_continuous_method, reduce_sum, cat_embed_cols, continuous_cols
):

    tab_preprocessor = TabPreprocessor(
        cat_embed_cols=cat_embed_cols,
        continuous_cols=continuous_cols,
        for_mf=True,
    )

    X_tab = tab_preprocessor.fit_transform(train)
    X_tab_tnsr = torch.tensor(X_tab)

    if embed_continuous_method == "periodic":
        model = DeepFactorizationMachine(
            column_idx=tab_preprocessor.column_idx,
            num_factors=8,
            reduce_sum=True,
            cat_embed_input=tab_preprocessor.cat_embed_input,
            continuous_cols=continuous_cols,
            embed_continuous_method=embed_continuous_method,
            n_frequencies=5,
            sigma=0.1,
            share_last_layer=False,
            mlp_hidden_dims=[16, 8],
        )
    else:
        quantization_setup = {
            "item_price": list(train["item_price"].quantile([0.25, 0.5, 0.75]).values),
            "user_age": list(train["user_age"].quantile([0.25, 0.5, 0.75]).values),
        }
        model = DeepFactorizationMachine(
            column_idx=tab_preprocessor.column_idx,
            num_factors=8,
            reduce_sum=True,
            cat_embed_input=tab_preprocessor.cat_embed_input,
            continuous_cols=continuous_cols,
            embed_continuous_method=embed_continuous_method,
            quantization_setup=quantization_setup,
            mlp_hidden_dims=[16, 8],
        )

    res = model(X_tab_tnsr)

    if reduce_sum:
        assert res.shape == (X_tab_tnsr.shape[0], 1)
    else:
        assert res.shape == (X_tab_tnsr.shape[0], model.output_dim)


def test_deepfm_model(cat_embed_cols, continuous_cols):

    tab_preprocessor = TabPreprocessor(
        cat_embed_cols=cat_embed_cols,
        continuous_cols=continuous_cols,
        for_mf=True,
    )

    X_tab = tab_preprocessor.fit_transform(train)
    X_tab_tnsr = torch.tensor(X_tab)

    wide_preprocessor = WidePreprocessor(wide_cols=cat_embed_cols + continuous_cols)
    # This is not entirely correct and needs comments in the docs
    train_for_wide = train.copy()
    train_for_wide["item_price"] = train_for_wide["item_price"].astype("int")
    train_for_wide["user_age"] = train_for_wide["user_age"].astype("int")
    X_wide = wide_preprocessor.fit_transform(train_for_wide)
    X_wide_tnsr = torch.tensor(X_wide)

    deepfm = DeepFactorizationMachine(
        column_idx=tab_preprocessor.column_idx,
        num_factors=8,
        reduce_sum=True,
        cat_embed_input=tab_preprocessor.cat_embed_input,
        continuous_cols=continuous_cols,
        embed_continuous_method="periodic",
        n_frequencies=5,
        sigma=0.1,
        share_last_layer=False,
        mlp_hidden_dims=[16, 8],
    )

    linear = Wide(input_dim=X_wide.max())

    fm_model = WideDeep(wide=linear, deeptabular=deepfm)

    X_inp = {"wide": X_wide_tnsr, "deeptabular": X_tab_tnsr}

    out = fm_model(X_inp)

    assert out.shape[0] == X_tab_tnsr.shape[0]
    assert out.shape[1] == 1


def test_deepfm_full_process(cat_embed_cols, continuous_cols):

    tab_preprocessor = TabPreprocessor(
        cat_embed_cols=cat_embed_cols,
        continuous_cols=continuous_cols,
        for_mf=True,
    )

    X_tab_tr = tab_preprocessor.fit_transform(train)
    y_tr = train["purchased"].values
    X_tab_val = tab_preprocessor.transform(valid)
    y_val = valid["purchased"].values
    X_tab_te = tab_preprocessor.transform(test)

    wide_preprocessor = WidePreprocessor(wide_cols=cat_embed_cols + continuous_cols)
    # This is not entirely correct and needs comments in the docs
    train_for_wide = train.copy()
    train_for_wide["item_price"] = train_for_wide["item_price"].astype("int")
    train_for_wide["user_age"] = train_for_wide["user_age"].astype("int")
    X_wide_tr = wide_preprocessor.fit_transform(train_for_wide)
    X_wide_val = wide_preprocessor.transform(valid)
    X_wide_te = wide_preprocessor.transform(test)

    deepfm = DeepFactorizationMachine(
        column_idx=tab_preprocessor.column_idx,
        num_factors=8,
        reduce_sum=True,
        cat_embed_input=tab_preprocessor.cat_embed_input,
        continuous_cols=continuous_cols,
        embed_continuous_method="periodic",
        n_frequencies=5,
        sigma=0.1,
        share_last_layer=False,
        mlp_hidden_dims=[16, 8],
    )

    linear = Wide(input_dim=X_wide_tr.max())

    fm_model = WideDeep(wide=linear, deeptabular=deepfm)

    trainer = Trainer(model=fm_model, objective="binary", verbose=0)

    X_train = {"X_wide": X_wide_tr, "X_tab": X_tab_tr, "target": y_tr}
    X_val = {"X_wide": X_wide_val, "X_tab": X_tab_val, "target": y_val}
    X_test = {"X_wide": X_wide_te, "X_tab": X_tab_te}

    trainer.fit(X_train=X_train, X_val=X_val, n_epochs=1)
    preds = trainer.predict(X_test=X_test)

    assert preds.shape[0] == X_tab_te.shape[0]
    assert (
        trainer.history is not None
        and "train_loss" in trainer.history
        and "val_loss" in trainer.history
    )
