import torch
import pytest

from pytorch_widedeep import Trainer
from pytorch_widedeep.models import Wide, WideDeep
from pytorch_widedeep.models.rec import DeepFieldAwareFactorizationMachine
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
def test_deepffm_reduce_sum(reduce_sum, mlp_hidden_dims, cat_embed_cols):

    tab_preprocessor = TabPreprocessor(
        cat_embed_cols=cat_embed_cols,
        for_mf=True,
    )

    X_tab = tab_preprocessor.fit_transform(train)

    deepffm = DeepFieldAwareFactorizationMachine(
        column_idx=tab_preprocessor.column_idx,
        num_factors=8,
        reduce_sum=reduce_sum,
        cat_embed_input=tab_preprocessor.cat_embed_input,
        mlp_hidden_dims=mlp_hidden_dims,
    )

    X_tab_tnsr = torch.tensor(X_tab)
    res = deepffm(X_tab_tnsr)

    if reduce_sum:
        assert res.shape == (X_tab_tnsr.shape[0], 1)
    else:
        assert res.shape == (X_tab_tnsr.shape[0], deepffm.output_dim)


def test_deepffm_model(cat_embed_cols):

    tab_preprocessor = TabPreprocessor(
        cat_embed_cols=cat_embed_cols,
        for_mf=True,
    )

    X_tab = tab_preprocessor.fit_transform(train)
    X_tab_tnsr = torch.tensor(X_tab)

    wide_preprocessor = WidePreprocessor(wide_cols=cat_embed_cols)
    X_wide = wide_preprocessor.fit_transform(train)
    X_wide_tnsr = torch.tensor(X_wide)

    deepffm = DeepFieldAwareFactorizationMachine(
        column_idx=tab_preprocessor.column_idx,
        num_factors=8,
        reduce_sum=True,
        cat_embed_input=tab_preprocessor.cat_embed_input,
        mlp_hidden_dims=[16, 8],
    )

    linear = Wide(input_dim=X_wide.max())

    fm_model = WideDeep(wide=linear, deeptabular=deepffm)

    X_inp = {"wide": X_wide_tnsr, "deeptabular": X_tab_tnsr}

    out = fm_model(X_inp)

    assert out.shape[0] == X_tab_tnsr.shape[0]
    assert out.shape[1] == 1


def test_deepffm_full_process(cat_embed_cols):

    tab_preprocessor = TabPreprocessor(
        cat_embed_cols=cat_embed_cols,
        for_mf=True,
    )

    X_tab_tr = tab_preprocessor.fit_transform(train)
    X_tab_val = tab_preprocessor.transform(valid)
    X_tab_te = tab_preprocessor.transform(test)

    y_tr = train["purchased"].values
    y_val = valid["purchased"].values

    wide_preprocessor = WidePreprocessor(wide_cols=cat_embed_cols)
    X_wide_tr = wide_preprocessor.fit_transform(train)
    X_wide_val = wide_preprocessor.transform(valid)
    X_wide_te = wide_preprocessor.transform(test)

    deepffm = DeepFieldAwareFactorizationMachine(
        column_idx=tab_preprocessor.column_idx,
        num_factors=8,
        reduce_sum=True,
        cat_embed_input=tab_preprocessor.cat_embed_input,
        mlp_hidden_dims=[16, 8],
    )

    linear = Wide(input_dim=X_wide_tr.max())

    fm_model = WideDeep(wide=linear, deeptabular=deepffm)

    X_train = {"X_wide": X_wide_tr, "X_tab": X_tab_tr, "target": y_tr}
    X_val = {"X_wide": X_wide_val, "X_tab": X_tab_val, "target": y_val}
    X_test = {"X_wide": X_wide_te, "X_tab": X_tab_te}

    trainer = Trainer(model=fm_model, objective="binary", verbose=0)

    trainer.fit(X_train=X_train, X_val=X_val, n_epochs=1)

    preds = trainer.predict(X_test=X_test)

    assert preds.shape[0] == X_tab_te.shape[0]
    assert (
        trainer.history is not None
        and "train_loss" in trainer.history
        and "val_loss" in trainer.history
    )
