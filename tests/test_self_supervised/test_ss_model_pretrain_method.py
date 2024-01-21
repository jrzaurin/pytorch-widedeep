import string

import numpy as np
import pandas as pd
import pytest

from pytorch_widedeep.models import TabMlp as TabMlpEncoder
from pytorch_widedeep.models import TabNet as TabNetEncoder
from pytorch_widedeep.models import TabResnet as TabResnetEncoder
from pytorch_widedeep.models import (
    TabMlpDecoder,
    TabNetDecoder,
    TabResnetDecoder,
)
from pytorch_widedeep.preprocessing import TabPreprocessor
from pytorch_widedeep.self_supervised_training import (
    EncoderDecoderTrainer,
    ContrastiveDenoisingTrainer,
)
from tests.test_self_supervised.test_ss_model_components import (
    _build_transf_model,
)

some_letters = list(string.ascii_lowercase)
some_numbers = range(10)
test_df = pd.DataFrame(
    {
        "col1": list(np.random.choice(some_letters, 32)),
        "col2": list(np.random.choice(some_letters, 32)),
        "col3": list(np.random.choice(some_numbers, 32)),
        "col4": list(np.random.choice(some_numbers, 32)),
    }
)


###############################################################################
# Test simply that the EncoderDecoderTrainer runs 'correctly'
###############################################################################


def _build_enc_models(model_type, column_idx, cat_embed_input, continuous_cols):
    if model_type == "mlp":
        encoder = TabMlpEncoder(
            column_idx=column_idx,
            cat_embed_input=cat_embed_input,
            continuous_cols=continuous_cols,
            mlp_hidden_dims=[16, 8],
        )

    if model_type == "resnet":
        encoder = TabResnetEncoder(
            column_idx=column_idx,
            cat_embed_input=cat_embed_input,
            continuous_cols=continuous_cols,
            blocks_dims=[32, 16, 8],
        )

    if model_type == "tabnet":
        encoder = TabNetEncoder(
            column_idx=column_idx,
            cat_embed_input=cat_embed_input,
            continuous_cols=continuous_cols,
        )

    return encoder


def _build_dec_models(model_type, encoder):
    if model_type == "mlp":
        decoder = TabMlpDecoder(
            embed_dim=encoder.cat_out_dim + encoder.cont_out_dim,
            mlp_hidden_dims=[encoder.output_dim, encoder.output_dim * 2],
        )

    if model_type == "resnet":
        decoder = TabResnetDecoder(
            embed_dim=encoder.cat_out_dim + encoder.cont_out_dim,
            blocks_dims=[
                encoder.output_dim,
                encoder.output_dim * 2,
                encoder.output_dim * 4,
            ],
        )

    if model_type == "tabnet":
        decoder = TabNetDecoder(
            embed_dim=encoder.cat_out_dim + encoder.cont_out_dim,
        )

    return decoder


@pytest.mark.parametrize(
    "model_type",
    ["mlp", "resnet", "tabnet"],
)
@pytest.mark.parametrize(
    "cat_or_cont",
    ["cat", "cont", "both"],
)
@pytest.mark.parametrize(
    "decoder_model",
    ["custom", "auto"],
)
def test_enc_dec_trainer(model_type, cat_or_cont, decoder_model):
    cat_embed_cols = ["col1", "col2"] if cat_or_cont in ["cat", "both"] else None
    continuous_cols = ["col3", "col4"] if cat_or_cont in ["cont", "both"] else None

    preprocessor = TabPreprocessor(
        cat_embed_cols=cat_embed_cols,
        continuous_cols=continuous_cols,
    )
    X_tab = preprocessor.fit_transform(test_df)

    cat_embed_input = (
        preprocessor.cat_embed_input
        if hasattr(preprocessor, "cat_embed_input")
        else None
    )

    encoder = _build_enc_models(
        model_type, preprocessor.column_idx, cat_embed_input, continuous_cols
    )

    if decoder_model == "auto":
        decoder = None
    elif decoder_model == "custom":
        decoder = _build_dec_models(model_type, encoder)

    ec_trainer = EncoderDecoderTrainer(
        encoder=encoder,
        decoder=decoder,
        masked_prob=0.2,
        verbose=0,
    )
    ec_trainer.pretrain(X_tab, n_epochs=2, batch_size=16)

    assert len(ec_trainer.history["train_loss"]) == 2


@pytest.mark.parametrize(
    "method_name",
    ["pretrain", "fit"],
)
def test_enc_dec_trainer_method_name(method_name):
    cat_embed_cols = ["col1", "col2"]
    continuous_cols = ["col3", "col4"]

    preprocessor = TabPreprocessor(
        cat_embed_cols=cat_embed_cols,
        continuous_cols=continuous_cols,
    )
    X_tab = preprocessor.fit_transform(test_df)

    encoder = _build_enc_models(
        "mlp",
        preprocessor.column_idx,
        preprocessor.cat_embed_input,
        preprocessor.continuous_cols,
    )

    ec_trainer = EncoderDecoderTrainer(
        encoder=encoder,
        masked_prob=0.2,
        verbose=0,
    )

    if method_name == "pretrain":
        ec_trainer.pretrain(X_tab, n_epochs=2, batch_size=16)
    elif method_name == "fit":
        ec_trainer.fit(X_tab, n_epochs=2, batch_size=16)

    assert len(ec_trainer.history["train_loss"]) == 2


###############################################################################
# Test simply that the ContrastiveDenoisingTrainer runs 'correctly'
###############################################################################


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
def test_cont_den_trainer_with_defaults(
    transf_model,
    cat_or_cont,
    with_cls_token,
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

    cat_embed_input = (
        preprocessor.cat_embed_input
        if hasattr(preprocessor, "cat_embed_input")
        else None
    )

    tr_model = _build_transf_model(
        transf_model, preprocessor, cat_embed_input, continuous_cols
    )

    cd_trainer = ContrastiveDenoisingTrainer(
        model=tr_model,
        preprocessor=preprocessor,
        verbose=0,
    )

    cd_trainer.pretrain(X_tab, n_epochs=2, batch_size=16)

    assert len(cd_trainer.history["train_loss"]) == 2


@pytest.mark.parametrize(
    "method_name",
    ["pretrain", "fit"],
)
def test_cont_den_trainer_method_name(
    method_name,
):
    cat_embed_cols = ["col1", "col2"]
    continuous_cols = ["col3", "col4"]

    preprocessor = TabPreprocessor(
        cat_embed_cols=cat_embed_cols,
        continuous_cols=continuous_cols,
        with_attention=True,
        with_cls_token=True,
    )
    X_tab = preprocessor.fit_transform(test_df)

    tr_model = _build_transf_model(
        "tabtransformer",
        preprocessor,
        preprocessor.cat_embed_input,
        preprocessor.continuous_cols,
    )

    cd_trainer = ContrastiveDenoisingTrainer(
        model=tr_model,
        preprocessor=preprocessor,
        verbose=0,
    )

    if method_name == "pretrain":
        cd_trainer.pretrain(X_tab, n_epochs=2, batch_size=16)
    elif method_name == "fit":
        cd_trainer.fit(X_tab, n_epochs=2, batch_size=16)

    assert len(cd_trainer.history["train_loss"]) == 2


###############################################################################
# Test that ContrastiveDenoisingTrainer with varying params
###############################################################################


@pytest.mark.parametrize(
    "loss_type",
    [
        "contrastive",
        "denoising",
        "both",
    ],
)
@pytest.mark.parametrize(
    "proj_head_dims",
    [None, [32, 8]],
)
@pytest.mark.parametrize(
    "mlp_type",
    ["single", "multiple"],
)
@pytest.mark.parametrize(
    "with_cls_token",
    [True, False],
)
def test_cont_den_trainer_with_varying_params(
    loss_type,
    proj_head_dims,
    mlp_type,
    with_cls_token,
):
    cat_embed_cols = ["col1", "col2"]
    continuous_cols = ["col3", "col4"]
    preprocessor = TabPreprocessor(
        cat_embed_cols=cat_embed_cols,
        continuous_cols=continuous_cols,
        with_attention=True,
        with_cls_token=with_cls_token,
    )
    X_tab = preprocessor.fit_transform(test_df)

    cat_embed_input = (
        preprocessor.cat_embed_input
        if hasattr(preprocessor, "cat_embed_input")
        else None
    )

    tr_model = _build_transf_model(
        "saint", preprocessor, cat_embed_input, continuous_cols
    )

    cd_trainer = ContrastiveDenoisingTrainer(
        model=tr_model,
        preprocessor=preprocessor,
        loss_type=loss_type,
        projection_head1_dims=proj_head_dims,
        projection_head2_dims=proj_head_dims,
        cat_mlp_type=mlp_type,
        cont_mlp_type=mlp_type,
        verbose=0,
    )

    cd_trainer.pretrain(X_tab, n_epochs=2, batch_size=16)

    assert len(cd_trainer.history["train_loss"]) == 2


@pytest.mark.parametrize(
    "proj_head_dims", [[None, [16, 8]], [[16, 8], None], [[16, 8], [16, 8]]]
)
def test_projection_head_value_error(
    proj_head_dims,
):
    cat_embed_cols = ["col1", "col2"]
    continuous_cols = ["col3", "col4"]
    preprocessor = TabPreprocessor(
        cat_embed_cols=cat_embed_cols,
        continuous_cols=continuous_cols,
        with_attention=True,
        with_cls_token=True,
    )
    X_tab = preprocessor.fit_transform(test_df)  # noqa: F841

    tr_model = _build_transf_model(
        "saint",
        preprocessor,
        preprocessor.cat_embed_input,
        preprocessor.continuous_cols,
    )

    with pytest.raises(ValueError):
        cd_trainer = ContrastiveDenoisingTrainer(  # noqa: F841
            model=tr_model,
            preprocessor=preprocessor,
            projection_head1_dims=proj_head_dims[0],
            projection_head2_dims=proj_head_dims[1],
            verbose=0,
        )
