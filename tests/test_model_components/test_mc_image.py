import numpy as np
import torch
import pytest

from pytorch_widedeep.models import Vision

X_images = (torch.from_numpy(np.random.rand(10, 3, 224, 224))).float()


###############################################################################
# Simply testing that it runs with the defaults
###############################################################################
def test_output_sizes():
    model = Vision()
    out = model(X_images)
    assert out.size(0) == 10 and out.size(1) == 512


###############################################################################
# Testing freeze_n
###############################################################################


def test_n_trainable():
    model = Vision(pretrained_model_name="resnet18", n_trainable=6)
    out = model(X_images)
    assert out.size(0) == 10 and out.size(1) == 512


###############################################################################
# Testing some the available architectures
###############################################################################


@pytest.mark.parametrize(
    "arch, expected_out_shape",
    [
        ("shufflenet_v2_x0_5", 1024),
        ("resnext50_32x4d", 2048),
        ("wide_resnet50_2", 2048),
        ("mobilenet_v2", 1280),
        ("mnasnet1_0", 1280),
        ("squeezenet1_0", 512),
    ],
)
def test_archiectures(arch, expected_out_shape):
    model = Vision(pretrained_model_name=arch, n_trainable=0)
    out = model(X_images)
    assert out.size(0) == 10 and out.size(1) == expected_out_shape


###############################################################################
# Testing the head
###############################################################################


def test_head():
    model = Vision(head_hidden_dims=[256, 128], head_dropout=0.1)
    out = model(X_images)
    assert out.size(0) == 10 and out.size(1) == 128


###############################################################################
# Make sure is all frozen
###############################################################################


def test_all_frozen():
    model = Vision(pretrained_model_name="resnet18", n_trainable=0)
    is_trainable = []
    for p in model.parameters():
        is_trainable.append(not p.requires_grad)
    assert all(is_trainable)
