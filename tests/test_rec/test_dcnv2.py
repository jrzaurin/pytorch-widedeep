import torch
import pytest

from pytorch_widedeep import Trainer
from pytorch_widedeep.models import WideDeep
from pytorch_widedeep.preprocessing import TabPreprocessor
from pytorch_widedeep.models.rec.dcnv2 import (
    CrossLayerV2,
    CrossNetworkV2,
    DeepCrossNetworkV2,
)

from .utils_test_rec import create_train_val_test_data

train, valid, test = create_train_val_test_data()


@pytest.mark.parametrize(
    "input_dim,low_rank,num_experts,expert_dropout",
    [
        (16, None, 1, 0.0),  # Basic case without low-rank
        (16, 8, 1, 0.0),  # With low-rank
        (16, None, 4, 0.1),  # Multiple experts
        (16, 8, 4, 0.1),  # Low-rank with multiple experts
    ],
)
def test_cross_layer_v2(input_dim, low_rank, num_experts, expert_dropout):
    batch_size = 32
    x = torch.randn(batch_size, input_dim)

    layer = CrossLayerV2(
        input_dim=input_dim,
        low_rank=low_rank,
        num_experts=num_experts,
        expert_dropout=expert_dropout,
    )

    # Test forward pass
    output = layer(x, x)
    assert output.shape == (batch_size, input_dim)

    # Test expert output shape
    expert_output = layer.get_expert_output(x)
    assert expert_output.shape == (batch_size, num_experts, input_dim)

    if num_experts > 1:
        assert hasattr(layer, "expert_gate")
        gate_output = layer.expert_gate(x)
        assert gate_output.shape == (batch_size, num_experts)


@pytest.mark.parametrize(
    "input_dim,num_layers,low_rank,num_experts",
    [
        (16, 2, None, 1),  # Basic case
        (16, 3, 8, 1),  # With low-rank
        (16, 2, None, 4),  # Multiple experts
        (16, 3, 8, 4),  # Low-rank with multiple experts
    ],
)
def test_cross_network_v2(input_dim, num_layers, low_rank, num_experts):
    batch_size = 32
    x = torch.randn(batch_size, input_dim)

    network = CrossNetworkV2(
        input_dim=input_dim,
        num_layers=num_layers,
        low_rank=low_rank,
        num_experts=num_experts,
    )

    # Test network structure
    assert len(network.cross_layers) == num_layers

    # Test forward pass
    output = network(x)
    assert output.shape == (batch_size, input_dim)


@pytest.mark.parametrize(
    "structure,num_experts,low_rank",
    [
        ("parallel", 1, None),  # Basic parallel
        ("stacked", 1, None),  # Basic stacked
        ("parallel", 2, 4),  # Parallel with experts and low-rank
        ("stacked", 2, 4),  # Stacked with experts and low-rank
    ],
)
def test_dcn_v2_structure(
    structure, num_experts, low_rank, cat_embed_cols, continuous_cols
):
    tab_preprocessor = TabPreprocessor(
        cat_embed_cols=cat_embed_cols,
        continuous_cols=continuous_cols,
    )

    X_tab = tab_preprocessor.fit_transform(train)
    X_tab_tnsr = torch.tensor(X_tab, dtype=torch.float32)

    model = DeepCrossNetworkV2(
        column_idx=tab_preprocessor.column_idx,
        low_rank=low_rank,
        num_experts=num_experts,
        structure=structure,
        cat_embed_input=tab_preprocessor.cat_embed_input,  # type: ignore[arg-type]
        continuous_cols=continuous_cols,
        mlp_hidden_dims=[16, 8],
    )

    out = model(X_tab_tnsr)
    assert out.shape == (X_tab_tnsr.shape[0], model.output_dim)

    # Test output dimension property
    if structure == "stacked":
        assert model.output_dim == model.mlp_hidden_dims[-1]
    else:  # parallel
        assert model.output_dim == (
            model.mlp_hidden_dims[-1] + model.cat_out_dim + model.cont_out_dim
        )


def test_dcn_v2_full_process(cat_embed_cols, continuous_cols):
    tab_preprocessor = TabPreprocessor(
        cat_embed_cols=cat_embed_cols,
        continuous_cols=continuous_cols,
    )

    X_tab_tr = tab_preprocessor.fit_transform(train)
    X_tab_val = tab_preprocessor.transform(valid)
    X_tab_te = tab_preprocessor.transform(test)
    y_tr = train["purchased"].values
    y_val = valid["purchased"].values

    dcn = DeepCrossNetworkV2(
        column_idx=tab_preprocessor.column_idx,
        num_cross_layers=2,
        num_experts=2,
        low_rank=4,
        cat_embed_input=tab_preprocessor.cat_embed_input,  # type: ignore[arg-type]
        continuous_cols=continuous_cols,
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
