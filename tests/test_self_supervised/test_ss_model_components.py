import string

import numpy as np
import torch
import pandas as pd
import pytest

from pytorch_widedeep.models import (
    SAINT,
    TabMlp,
    TabNet,
    TabResnet,
    FTTransformer,
    TabFastFormer,
    TabMlpDecoder,
    TabNetDecoder,
    TabTransformer,
    SelfAttentionMLP,
    TabResnetDecoder,
    ContextAttentionMLP,
)
from pytorch_widedeep.preprocessing import TabPreprocessor
from pytorch_widedeep.models.tabular.self_supervised import (
    EncoderDecoderModel,
    ContrastiveDenoisingModel,
)

colnames = list(string.ascii_lowercase)[:10]
embed_cols = [np.random.choice(np.arange(5), 10) for _ in range(5)]
embed_input = [(u, i, j) for u, i, j in zip(colnames[:5], [5] * 5, [8] * 5)]
cont_cols = [np.random.rand(10) for _ in range(5)]
continuous_cols = colnames[-5:]

X_deep = torch.from_numpy(np.vstack(embed_cols + cont_cols).transpose())
X_deep_emb = X_deep[:, :5]
X_deep_cont = X_deep[:, 5:]
target = np.random.choice(2, 32)


@pytest.mark.parametrize(
    "model_type",
    ["mlp", "resnet", "tabnet"],
)
@pytest.mark.parametrize(
    "cat_or_cont",
    ["cat", "cont", "both"],
)
@pytest.mark.parametrize(
    "infer_decoder",
    [True, False],
)
def test_enc_dec_different_setups(model_type, cat_or_cont, infer_decoder):
    if cat_or_cont == "cat":
        column_idx = {k: v for v, k in enumerate(colnames[:5])}
        cont_cols = None
        X = X_deep_emb
    if cat_or_cont == "cont":
        column_idx = {k: v for v, k in enumerate(colnames[5:])}
        cont_cols = continuous_cols
        X = X_deep_cont
    if cat_or_cont == "both":
        column_idx = {k: v for v, k in enumerate(colnames)}
        cont_cols = continuous_cols
        X = X_deep

    if model_type == "mlp":
        encoder = TabMlp(
            column_idx=column_idx,
            cat_embed_input=embed_input if cat_or_cont == "cat" else None,
            continuous_cols=cont_cols,
            mlp_hidden_dims=[16, 8],
        )
        if infer_decoder:
            decoder = None
        else:
            # self.encoder.cat_out_dim + self.encoder.cont_out_dim
            embed_dim = encoder.cat_out_dim + encoder.cont_out_dim
            decoder = TabMlpDecoder(embed_dim=embed_dim, mlp_hidden_dims=[8, 16])
        ed_model = EncoderDecoderModel(encoder, decoder, 0.2)

    if model_type == "resnet":
        encoder = TabResnet(
            column_idx=column_idx,
            cat_embed_input=embed_input if cat_or_cont == "cat" else None,
            continuous_cols=cont_cols,
            blocks_dims=[16, 8, 8],
        )
        if infer_decoder:
            decoder = None
        else:
            embed_dim = encoder.cat_out_dim + encoder.cont_out_dim
            decoder = TabResnetDecoder(embed_dim=embed_dim, blocks_dims=[8, 8, 16])
        ed_model = EncoderDecoderModel(encoder, decoder, 0.2)

    if model_type == "tabnet":
        encoder = TabNet(
            column_idx=column_idx,
            cat_embed_input=embed_input if cat_or_cont == "cat" else None,
            continuous_cols=cont_cols,
        )
        if infer_decoder:
            decoder = None
        else:
            embed_dim = encoder.cat_out_dim + encoder.cont_out_dim
            decoder = TabNetDecoder(embed_dim=embed_dim)
        ed_model = EncoderDecoderModel(encoder, decoder, 0.2)

    out, out_rec, mask = ed_model(X)

    assert (
        out.size(1) == encoder.cat_out_dim + encoder.cont_out_dim
        and out.size() == out_rec.size()
        and (mask.sum() / mask.numel() <= 0.4).item()  # this last one is not ideal
    )


some_letters = list(string.ascii_lowercase)[:10]
some_numbers = range(10)
test_df = pd.DataFrame(
    {
        "col1": list(np.random.choice(some_letters, 10)),
        "col2": list(np.random.choice(some_letters, 10)),
        "col3": list(np.random.choice(some_numbers, 10)),
        "col4": list(np.random.choice(some_numbers, 10)),
    }
)


@pytest.mark.parametrize(
    "transf_model",
    [
        "tabtransformer",
        "saint",
        "fttransformer",
        "tabfastformer",
        "contextattentionmlp",
        "selfattentionmlp",
    ],
)
@pytest.mark.parametrize(
    "cat_or_cont",
    ["cat", "cont", "both"],
)
@pytest.mark.parametrize(
    "with_cls_token",
    [True, False],
)
def test_cont_den_multiple_mlps_different_setups(
    transf_model, cat_or_cont, with_cls_token
):
    cat_embed_cols = ["col1", "col2"] if cat_or_cont in ["cat", "both"] else None
    continuous_cols = ["col3", "col4"] if cat_or_cont in ["cont", "both"] else None

    preprocessor = TabPreprocessor(
        cat_embed_cols=cat_embed_cols,
        continuous_cols=continuous_cols,
        with_attention=True,
        with_cls_token=with_cls_token,
    )
    X_tab = preprocessor.fit_transform(test_df)
    X = torch.from_numpy(X_tab)

    cat_embed_input = (
        preprocessor.cat_embed_input
        if hasattr(preprocessor, "cat_embed_input")
        else None
    )

    tr_model = _build_transf_model(
        transf_model, preprocessor, cat_embed_input, continuous_cols
    )
    cd_model = ContrastiveDenoisingModel(
        tr_model,
        preprocessor,
        loss_type="both",
        projection_head1_dims=None,
        projection_head2_dims=None,
        projection_heads_activation="relu",
        cat_mlp_type="multiple",
        cont_mlp_type="multiple",
        denoise_mlps_activation="relu",
    )

    if cat_or_cont in ["cat", "both"]:
        out_dim = []
        for name, param in cd_model.denoise_cat_mlp.named_parameters():
            if ("dense_layer_1.0" in name) and ("weight" in name):
                out_dim.append(param.shape[0])

    g_projs, cat_x_and_x_, cont_x_and_x_ = cd_model(X)

    checks = []
    if g_projs is not None:
        projs_check = _check_g_projs(X, g_projs, tr_model, with_cls_token)

        checks.extend(projs_check)

    if cat_x_and_x_ is not None:
        cat_checks = _check_cat_multiple_denoise_mlps(
            X, cat_x_and_x_, with_cls_token, out_dim
        )

        checks.extend(cat_checks)

    if cat_or_cont == "both":
        cont_if_cat_check = _check_cont_if_cat_multiple_denoise_mlps(
            X, cont_x_and_x_, with_cls_token
        )

        checks.extend(cont_if_cat_check)

    elif cat_or_cont == "cont":
        cont_only_check = _check_cont_only_multiple_denoise_mlps(
            X, cont_x_and_x_, with_cls_token
        )

        checks.extend(cont_only_check)

    assert all(checks)


@pytest.mark.parametrize(
    "transf_model",
    [
        "tabtransformer",
        "saint",
        "fttransformer",
        "tabfastformer",
        "contextattentionmlp",
        "selfattentionmlp",
    ],
)
@pytest.mark.parametrize(
    "cat_or_cont",
    ["cat", "cont", "both"],
)
@pytest.mark.parametrize(
    "with_cls_token",
    [True, False],
)
def test_cont_den_single_mlp_different_setups(
    transf_model, cat_or_cont, with_cls_token
):
    cat_embed_cols = ["col1", "col2"] if cat_or_cont in ["cat", "both"] else None
    continuous_cols = ["col3", "col4"] if cat_or_cont in ["cont", "both"] else None

    preprocessor = TabPreprocessor(
        cat_embed_cols=cat_embed_cols,
        continuous_cols=continuous_cols,
        with_attention=True,
        with_cls_token=with_cls_token,
    )
    X_tab = preprocessor.fit_transform(test_df)
    X = torch.from_numpy(X_tab)

    cat_embed_input = (
        preprocessor.cat_embed_input
        if hasattr(preprocessor, "cat_embed_input")
        else None
    )

    tr_model = _build_transf_model(
        transf_model, preprocessor, cat_embed_input, continuous_cols
    )
    cd_model = ContrastiveDenoisingModel(
        tr_model,
        preprocessor,
        loss_type="both",
        projection_head1_dims=None,
        projection_head2_dims=None,
        projection_heads_activation="relu",
        cat_mlp_type="single",
        cont_mlp_type="single",
        denoise_mlps_activation="relu",
    )

    out_dim = _get_output_dim(cd_model) if cat_or_cont in ["cat", "both"] else None

    g_projs, cat_x_and_x_, cont_x_and_x_ = cd_model(X)

    checks = []
    if g_projs is not None:
        projs_check = _check_g_projs(X, g_projs, tr_model, with_cls_token)

        checks.extend(projs_check)

    if cat_x_and_x_ is not None:
        cat_checks = _check_cat_single_denoise_mlps(
            X, cat_x_and_x_, with_cls_token, out_dim
        )

        checks.extend(cat_checks)

    if cat_or_cont == "both":
        cont_if_cat_check = _check_cont_if_cat_single_denoise_mlps(
            X, cont_x_and_x_, with_cls_token
        )

        checks.extend(cont_if_cat_check)

    elif cat_or_cont == "cont":
        cont_only_check = _check_cont_only_single_denoise_mlps(
            X, cont_x_and_x_, with_cls_token
        )

        checks.extend(cont_only_check)

    assert all(checks)


def _get_output_dim(cd_model):
    for name, param in cd_model.denoise_cat_mlp.named_parameters():
        if ("dense_layer_1.0" in name) and ("weight" in name):
            out_dim = param.shape[0]
    return out_dim


def _check_cat_single_denoise_mlps(X, cat_x_and_x_, with_cls_token, out_dim):
    assertions = []

    targ = (
        torch.cat([X[:, 1], X[:, 2]]) - 2
        if with_cls_token
        else torch.cat([X[:, 0], X[:, 1]]) - 1
    )

    assrt1 = targ.shape[0] == X.shape[0] * 2
    assrt2 = all(targ == cat_x_and_x_[0])
    assrt3 = (targ.max() + 1).item() == out_dim

    assertions.extend([assrt1, assrt2, assrt3])

    return assertions


def _check_cont_if_cat_single_denoise_mlps(X, cont_x_and_x_, with_cls_token):
    assertions = []

    targ = (
        torch.cat([X[:, 3], X[:, 4]])
        if with_cls_token
        else torch.cat([X[:, 2], X[:, 3]])
    )

    assrt1 = all(torch.isclose(cont_x_and_x_[0].squeeze(1), targ.float()))

    assertions.extend([assrt1])

    return assertions


def _check_cont_only_single_denoise_mlps(X, cont_x_and_x_, with_cls_token):
    assertions = []

    targ = (
        torch.cat([X[:, 1], X[:, 2]])
        if with_cls_token
        else torch.cat([X[:, 0], X[:, 1]])
    )

    assrt1 = all(torch.isclose(cont_x_and_x_[0].squeeze(1), targ.float()))

    assertions.extend([assrt1])

    return assertions


def _build_transf_model(transf_model, preprocessor, cat_embed_input, continuous_cols):
    if transf_model == "tabtransformer":
        model = TabTransformer(
            column_idx=preprocessor.column_idx,
            cat_embed_input=cat_embed_input,
            continuous_cols=continuous_cols,
            embed_continuous=True,
            embed_continuous_method="standard",
            n_heads=2,
            n_blocks=2,
        )
    if transf_model == "saint":
        model = SAINT(
            column_idx=preprocessor.column_idx,
            cat_embed_input=cat_embed_input,
            continuous_cols=continuous_cols,
            n_blocks=2,
            n_heads=2,
        )
    if transf_model == "fttransformer":
        model = FTTransformer(
            column_idx=preprocessor.column_idx,
            cat_embed_input=cat_embed_input,
            continuous_cols=continuous_cols,
            n_blocks=2,
            n_heads=2,
        )
    if transf_model == "tabfastformer":
        model = TabFastFormer(
            column_idx=preprocessor.column_idx,
            cat_embed_input=cat_embed_input,
            continuous_cols=continuous_cols,
            n_blocks=2,
            n_heads=2,
        )
    if transf_model == "contextattentionmlp":
        model = ContextAttentionMLP(
            column_idx=preprocessor.column_idx,
            cat_embed_input=cat_embed_input,
            continuous_cols=continuous_cols,
        )
    if transf_model == "selfattentionmlp":
        model = SelfAttentionMLP(
            column_idx=preprocessor.column_idx,
            cat_embed_input=cat_embed_input,
            continuous_cols=continuous_cols,
        )

    return model


def _check_g_projs(X, g_projs, model, with_cls_token):
    assertions = []

    asrt1 = g_projs[0].shape[1] == X.shape[1] - 1 if with_cls_token else X.shape[1]
    asrt2 = g_projs[1].shape[1] == X.shape[1] - 1 if with_cls_token else X.shape[1]
    asrt3 = g_projs[0].shape[2] == model.input_dim
    asrt4 = g_projs[1].shape[2] == model.input_dim

    assertions.extend([asrt1, asrt2, asrt3, asrt4])

    return assertions


def _check_cat_multiple_denoise_mlps(X, cat_x_and_x_, with_cls_token, out_dim):
    assertions = []

    targ1 = (X[:, 1] - 2).long() if with_cls_token else (X[:, 0] - 1).long()
    idx_to_substract_col2 = min(X[:, 2]) if with_cls_token else min(X[:, 1])
    targ2 = (
        (X[:, 2] - idx_to_substract_col2).long()
        if with_cls_token
        else (X[:, 1] - idx_to_substract_col2).long()
    )

    assrt1 = all(cat_x_and_x_[0][0] == targ1)
    assrt2 = all(cat_x_and_x_[1][0] == targ2)
    assrt3 = cat_x_and_x_[0][1].shape[1] == out_dim[0]
    assrt4 = cat_x_and_x_[1][1].shape[1] == out_dim[1]

    assertions.extend([assrt1, assrt2, assrt3, assrt4])

    return assertions


def _check_cont_if_cat_multiple_denoise_mlps(X, cont_x_and_x_, with_cls_token):
    assertions = []

    targ1 = X[:, 3] if with_cls_token else X[:, 2]
    targ2 = X[:, 4] if with_cls_token else X[:, 3]

    assrt1 = all(torch.isclose(cont_x_and_x_[0][0].squeeze(1), targ1.float()))
    assrt2 = all(torch.isclose(cont_x_and_x_[1][0].squeeze(1), targ2.float()))

    assertions.extend([assrt1, assrt2])

    return assertions


def _check_cont_only_multiple_denoise_mlps(X, cont_x_and_x_, with_cls_token):
    assertions = []

    targ1 = X[:, 1] if with_cls_token else X[:, 0]
    targ2 = X[:, 2] if with_cls_token else X[:, 1]

    assrt1 = all(torch.isclose(cont_x_and_x_[0][0].squeeze(1), targ1.float()))
    assrt2 = all(torch.isclose(cont_x_and_x_[1][0].squeeze(1), targ2.float()))

    assertions.extend([assrt1, assrt2])

    return assertions
