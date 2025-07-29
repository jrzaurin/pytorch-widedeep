import string

import numpy as np
import torch
import pytest
from torch import nn
from torchvision.transforms import ToTensor, Normalize

from pytorch_widedeep.models import Wide, TabMlp, Vision, BasicRNN, WideDeep
from pytorch_widedeep.training import Trainer

# Check if multiple GPUs are available
multi_gpu_available = torch.cuda.is_available() and torch.cuda.device_count() > 1

# Set random seed for reproducibility
np.random.seed(42)

# Create small matrices (500 observations)
n_obs = 500

# Wide array
X_wide = np.random.choice(50, (n_obs, 10))

# Deep Array (Tabular)
colnames = list(string.ascii_lowercase)[:10]
embed_cols = [np.random.choice(np.arange(5), n_obs) for _ in range(5)]
embed_input = [(u, i, j) for u, i, j in zip(colnames[:5], [5] * 5, [16] * 5)]
cont_cols = [np.random.rand(n_obs) for _ in range(5)]
column_idx = {k: v for v, k in enumerate(colnames)}
X_tab = np.vstack(embed_cols + cont_cols).transpose()

# Text Array
padded_sequences = np.random.choice(np.arange(1, 100), (n_obs, 48))
X_text = np.hstack((np.repeat(np.array([[0, 0]]), n_obs, axis=0), padded_sequences))
vocab_size = 110

# Image Array
X_img = np.random.choice(256, (n_obs, 224, 224, 3))
X_img_norm = X_img / 255.0

# Target
target = np.random.choice(2, n_obs)


@pytest.fixture
def model_components():
    # Wide component
    wide = Wide(np.unique(X_wide).shape[0], 1)

    # Tabular component
    deeptabular = TabMlp(
        column_idx=column_idx,
        cat_embed_input=embed_input,
        continuous_cols=colnames[-5:],
        mlp_hidden_dims=[32, 16],
        mlp_dropout=[0.5, 0.5],
    )

    # Text component
    deeptext = BasicRNN(vocab_size=vocab_size, embed_dim=32, padding_idx=0)

    # Image component
    deepimage = Vision(pretrained_model_setup="resnet18", n_trainable=0)

    return wide, deeptabular, deeptext, deepimage


@pytest.mark.skipif(not multi_gpu_available, reason="Multiple GPUs not available")
def test_multi_gpu_all_components(model_components):
    wide, deeptabular, deeptext, deepimage = model_components

    model = WideDeep(
        wide=wide, deeptabular=deeptabular, deeptext=deeptext, deepimage=deepimage
    )

    trainer = Trainer(model, objective="binary", verbose=0, use_multi_gpu=True)

    trainer.fit(
        X_wide=X_wide,
        X_tab=X_tab,
        X_text=X_text,
        X_img=X_img,
        target=target,
        batch_size=32,
        n_epochs=1,
    )

    assert trainer.history["train_loss"] is not None


@pytest.mark.skipif(not multi_gpu_available, reason="Multiple GPUs not available")
def test_multi_gpu_deeptabular_only(model_components):
    _, deeptabular, _, _ = model_components

    model = WideDeep(deeptabular=deeptabular)

    trainer = Trainer(model, objective="binary", verbose=0, use_multi_gpu=True)

    trainer.fit(X_tab=X_tab, target=target, batch_size=32, n_epochs=1)

    assert trainer.history["train_loss"] is not None


@pytest.mark.skipif(not multi_gpu_available, reason="Multiple GPUs not available")
def test_multi_gpu_with_deephead(model_components):
    _, deeptabular, deeptext, deepimage = model_components

    # Create a deephead
    deephead = nn.Sequential(nn.Linear(592, 128), nn.ReLU(), nn.Linear(128, 64))
    deephead.output_dim = 64

    model = WideDeep(
        deeptabular=deeptabular,
        deeptext=deeptext,
        deepimage=deepimage,
        deephead=deephead,
    )

    trainer = Trainer(model, objective="binary", verbose=0, use_multi_gpu=True)

    trainer.fit(
        X_tab=X_tab,
        X_text=X_text,
        X_img=X_img,
        target=target,
        batch_size=32,
        n_epochs=1,
    )

    assert trainer.history["train_loss"] is not None


@pytest.mark.skipif(not multi_gpu_available, reason="Multiple GPUs not available")
def test_multi_gpu_with_transforms(model_components):
    _, deeptabular, _, deepimage = model_components

    # Define transforms
    mean = [0.406, 0.456, 0.485]  # BGR
    std = [0.225, 0.224, 0.229]  # BGR
    transforms = [ToTensor(), Normalize(mean=mean, std=std)]

    model = WideDeep(deeptabular=deeptabular, deepimage=deepimage)

    trainer = Trainer(
        model, objective="binary", transforms=transforms, verbose=0, use_multi_gpu=True
    )

    trainer.fit(X_tab=X_tab, X_img=X_img, target=target, batch_size=32, n_epochs=1)

    assert trainer.history["train_loss"] is not None


@pytest.mark.skipif(not multi_gpu_available, reason="Multiple GPUs not available")
def test_multi_gpu_with_finetune(model_components):
    _, deeptabular, _, _ = model_components

    model = WideDeep(deeptabular=deeptabular)

    trainer = Trainer(model, objective="binary", verbose=0, use_multi_gpu=True)

    trainer.fit(
        X_tab=X_tab,
        target=target,
        batch_size=32,
        n_epochs=1,
        finetune=True,
        finetune_epochs=1,
    )

    assert trainer.history["train_loss"] is not None
    assert hasattr(trainer, "with_finetuning")
    assert trainer.with_finetuning is True


@pytest.mark.skipif(not multi_gpu_available, reason="Multiple GPUs not available")
def test_multi_gpu_predict(model_components):
    _, deeptabular, _, _ = model_components

    model = WideDeep(deeptabular=deeptabular)

    trainer = Trainer(model, objective="binary", verbose=0, use_multi_gpu=True)

    trainer.fit(X_tab=X_tab, target=target, batch_size=32, n_epochs=1)

    preds = trainer.predict(X_tab=X_tab, batch_size=32)
    assert preds.shape[0] == n_obs

    probs = trainer.predict_proba(X_tab=X_tab, batch_size=32)
    assert probs.shape[0] == n_obs
    assert probs.shape[1] == 2
