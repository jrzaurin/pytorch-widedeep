import os

import numpy as np
import torch
import pytest
from torchvision.models import (
    MNASNet1_0_Weights,
    MobileNet_V2_Weights,
    SqueezeNet1_0_Weights,
    ResNeXt50_32X4D_Weights,
    Wide_ResNet50_2_Weights,
    ShuffleNet_V2_X0_5_Weights,
)

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
    model = Vision(pretrained_model_setup="resnet18", n_trainable=6)
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
        ({"shufflenet_v2_x0_5": ShuffleNet_V2_X0_5_Weights.IMAGENET1K_V1}, 1024),
        ({"resnext50_32x4d": ResNeXt50_32X4D_Weights.IMAGENET1K_V2}, 2048),
        ({"wide_resnet50_2": Wide_ResNet50_2_Weights.IMAGENET1K_V2}, 2048),
        ({"mobilenet_v2": MobileNet_V2_Weights.IMAGENET1K_V2}, 1280),
        ({"mnasnet1_0": MNASNet1_0_Weights.IMAGENET1K_V1}, 1280),
        ({"squeezenet1_0": SqueezeNet1_0_Weights.IMAGENET1K_V1}, 512),
    ],
)
def test_architectures(arch, expected_out_shape):
    model = Vision(pretrained_model_setup=arch, n_trainable=0)
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
    model = Vision(pretrained_model_setup="resnet18", n_trainable=0)
    is_trainable = []
    for p in model.parameters():
        is_trainable.append(not p.requires_grad)
    assert all(is_trainable)


###############################################################################
# Check defaulting for arch classes
###############################################################################


@pytest.mark.parametrize(
    "arch, expected_out_shape",
    [
        ("resnet", 512),
        ("shufflenet", 1024),
        ("resnext", 2048),
        ("wide_resnet", 2048),
        ("regnet", 912),
        ("mobilenet", 1280),
        ("mnasnet", 1280),
        # ("efficientnet", 1280),
        ("squeezenet", 512),
        ({"shufflenet": ShuffleNet_V2_X0_5_Weights.IMAGENET1K_V1}, 1024),
        ({"resnext": ResNeXt50_32X4D_Weights.IMAGENET1K_V2}, 2048),
    ],
)
def test_pretrained_model_setup_defaults(arch, expected_out_shape):
    model = Vision(pretrained_model_setup=arch, n_trainable=0)
    out = model(X_images)
    assert out.size(0) == 10 and out.size(1) == expected_out_shape


IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"


@pytest.mark.skipif(
    IN_GITHUB_ACTIONS,
    reason=(
        "For reasons beyond me, when running in GH actions, "
        "throws a RuntimeError when trying to download the weights"
    ),
)
def test_pretrained_model_efficientnet():
    model = Vision(pretrained_model_setup="efficientnet", n_trainable=0)
    out = model(X_images)
    assert out.size(0) == 10 and out.size(1) == 1280
