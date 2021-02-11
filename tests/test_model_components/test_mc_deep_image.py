import numpy as np
import torch
import pytest

from pytorch_widedeep.models import DeepImage

X_images = (torch.from_numpy(np.random.rand(10, 3, 224, 224))).float()


###############################################################################
# Simply testing that it runs with the defaults
###############################################################################
model1 = DeepImage()


def test_deep_image_1():
    out = model1(X_images)
    assert out.size(0) == 10 and out.size(1) == 512


###############################################################################
# Testing with 'custom' backbone
###############################################################################
model2 = DeepImage(pretrained=False)


def test_deep_image_custom_backbone():
    out = model2(X_images)
    assert out.size(0) == 10 and out.size(1) == 512


###############################################################################
# Testing freeze_n
###############################################################################
model4 = DeepImage(freeze_n=5)


def test_deep_image_freeze_int():
    out = model4(X_images)
    assert out.size(0) == 10 and out.size(1) == 512


###############################################################################
# Testing with resnet 34
###############################################################################
model5 = DeepImage(resnet_architecture=34)


def test_deep_image_resnet_34():
    out = model5(X_images)
    assert out.size(0) == 10 and out.size(1) == 512


###############################################################################
# Testing the head
###############################################################################
model6 = DeepImage(head_hidden_dims=[512, 256, 128], head_dropout=[0.0, 0.5])


def test_deep_image_2():
    out = model6(X_images)
    assert out.size(0) == 10 and out.size(1) == 128


###############################################################################
# Make sure is frozen
###############################################################################

model7 = DeepImage(freeze_n=8)


def test_all_frozen():
    is_trainable = []
    for p in model7.parameters():
        is_trainable.append(not p.requires_grad)
    assert all(is_trainable)


###############################################################################
# Catch Exception
###############################################################################


def test_too_cold():
    with pytest.raises(ValueError):
        mod = DeepImage(freeze_n=10)  # noqa: F841
