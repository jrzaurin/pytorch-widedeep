import torch
import pytest

from pytorch_widedeep import Trainer
from pytorch_widedeep.models import WideDeep
from pytorch_widedeep.preprocessing import TabPreprocessor
from pytorch_widedeep.models.rec.gdcn import (
    GatedCrossLayer,
    GatedCrossNetwork,
    GatedDeepCrossNetwork,
)

from .utils_test_rec import create_train_val_test_data

train, valid, test = create_train_val_test_data()


def test_gated_cross_layer():
    input_dim = 16
    batch_size = 32

    layer = GatedCrossLayer(input_dim=input_dim)

    # Test input tensors
    x0 = torch.randn(batch_size, input_dim)
    xi = torch.randn(batch_size, input_dim)

    # Test forward pass
    output = layer(x0, xi)

    # Check output shape
    assert output.shape == (batch_size, input_dim)

    # Check parameter shapes
    assert layer.cross_weight.shape == (input_dim, input_dim)
    assert layer.gate_weight.shape == (input_dim, input_dim)
    assert layer.bias.shape == (input_dim,)


@pytest.mark.parametrize("num_layers", [2, 3, 4])
def test_gated_cross_network(num_layers):
    input_dim = 16
    batch_size = 32

    network = GatedCrossNetwork(input_dim=input_dim, num_layers=num_layers)

    # Test input tensor
    x = torch.randn(batch_size, input_dim)

    # Test forward pass
    output = network(x)

    # Check output shape
    assert output.shape == (batch_size, input_dim)

    # Check number of layers
    assert len(network.layers) == num_layers

    # Check each layer is a GatedCrossLayer
    for layer in network.layers:
        assert isinstance(layer, GatedCrossLayer)


@pytest.mark.parametrize(
    "structure,num_layers",
    [
        ("parallel", 2),
        ("stacked", 2),
        ("parallel", 3),
        ("stacked", 3),
    ],
)
def test_gdcn_structure(structure, num_layers, cat_embed_cols, continuous_cols):
    tab_preprocessor = TabPreprocessor(
        cat_embed_cols=cat_embed_cols, continuous_cols=continuous_cols
    )

    X_tab = tab_preprocessor.fit_transform(train)
    X_tab_tnsr = torch.tensor(X_tab, dtype=torch.float32)

    model = GatedDeepCrossNetwork(
        column_idx=tab_preprocessor.column_idx,
        num_cross_layers=num_layers,
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


def test_gdcn_full_process(cat_embed_cols):
    tab_preprocessor = TabPreprocessor(
        cat_embed_cols=cat_embed_cols,
    )

    X_tab_tr = tab_preprocessor.fit_transform(train)
    X_tab_val = tab_preprocessor.transform(valid)
    X_tab_te = tab_preprocessor.transform(test)
    y_tr = train["purchased"].values
    y_val = valid["purchased"].values

    gdcn = GatedDeepCrossNetwork(
        column_idx=tab_preprocessor.column_idx,
        cat_embed_input=tab_preprocessor.cat_embed_input,  # type: ignore[arg-type]
        num_cross_layers=2,
        mlp_hidden_dims=[16, 8],
    )
    model = WideDeep(deeptabular=gdcn)

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
