import torch
import pytest

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
