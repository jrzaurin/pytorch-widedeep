import numpy as np
import torch
import pytest

from pytorch_widedeep.models.deep_image import DeepImage

X_images = (torch.from_numpy(np.random.rand(10, 3, 224, 224))).float()

###############################################################################
# Simply testing that it runs with the defaults
###############################################################################
model1 = DeepImage()
def test_deep_image():
	out = model1(X_images)
	assert out.size(0) == 10 and out.size(1) == 1


###############################################################################
# Testing with custome backbone
###############################################################################
model2 = DeepImage(pretrained=False)
def test_deep_image_custom_backbone():
	out = model2(X_images)
	assert out.size(0) == 10 and out.size(1) == 1


###############################################################################
# Testing with freeze='all'
###############################################################################
model3 = DeepImage(freeze='all')
def test_deep_image_freeze_all():
	out = model3(X_images)
	assert out.size(0) == 10 and out.size(1) == 1


###############################################################################
# Testing with freeze=int
###############################################################################
model4 = DeepImage(freeze=5)
def test_deep_image_freeze_int():
	out = model4(X_images)
	assert out.size(0) == 10 and out.size(1) == 1


###############################################################################
# Testing with resnet 34
###############################################################################
model5 = DeepImage(resnet=34)
def test_deep_image_resnet_34():
	out = model5(X_images)
	assert out.size(0) == 10 and out.size(1) == 1
