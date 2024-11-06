import os

import cv2
import numpy as np
import pandas as pd
import pytest

from pytorch_widedeep.utils import SimplePreprocessor, AspectAwarePreprocessor
from pytorch_widedeep.preprocessing import ImagePreprocessor
from pytorch_widedeep.utils.image_utils import resize

full_path = os.path.realpath(__file__)
path = os.path.split(full_path)[0]
df = pd.DataFrame({"galaxies": ["galaxy1.png", "galaxy2.png"]})
img_col = "galaxies"
imd_dir = os.path.join(path, "images")
processor = ImagePreprocessor(img_col=img_col, img_path=imd_dir)
X_imgs = processor.fit_transform(df)

###############################################################################
# test on the 2 preprocessors directly
###############################################################################


def test_aap_ssp():
    img = cv2.imread("/".join([imd_dir, "galaxy1.png"]))

    aap = AspectAwarePreprocessor(128, 128)
    spp = SimplePreprocessor(128, 128)

    out1 = aap.preprocess(img)
    out2 = aap.preprocess(img.transpose(1, 0, 2))
    out3 = spp.preprocess(img)

    assert out1.shape[0] == 128 and out2.shape[0] == 128 and out3.shape[0] == 128


###############################################################################
# There is not much to test here, since I only resize.
###############################################################################


def test_sizes():
    img_width = X_imgs.shape[1]
    img_height = X_imgs.shape[2]
    assert np.all((img_width == processor.width, img_height == processor.height))


###############################################################################
# Test NotImplementedError in inverse transform
###############################################################################


def test_notimplementederror():
    with pytest.raises(NotImplementedError):
        org_df = processor.inverse_transform(X_imgs)  # noqa: F841


###############################################################################
# testthe resize function
###############################################################################


@pytest.fixture
def sample_image():
    # Create a sample 100x200 RGB image
    return np.zeros((100, 200, 3), dtype=np.uint8)


def test_resize_width_only(sample_image):
    # Test resizing with only width specified
    width = 100
    resized = resize(sample_image, width=width)

    # Check dimensions
    assert resized.shape[1] == width  # width should match
    assert resized.shape[2] == 3  # channels should remain unchanged
    # Height should be proportionally scaled (50 in this case)
    assert resized.shape[0] == 50


def test_resize_height_only(sample_image):
    # Test resizing with only height specified
    height = 50
    resized = resize(sample_image, height=height)

    # Check dimensions
    assert resized.shape[0] == height  # height should match
    assert resized.shape[2] == 3  # channels should remain unchanged
    # Width should be proportionally scaled (100 in this case)
    assert resized.shape[1] == 100


def test_resize_none_dimensions(sample_image):
    # Test when both width and height are None
    resized = resize(sample_image, width=None, height=None)

    # Image should remain unchanged
    assert np.array_equal(resized, sample_image)
    assert resized.shape == sample_image.shape


def test_resize_different_interpolation(sample_image):
    # Test with different interpolation method
    width = 100
    resized = resize(sample_image, width=width, inter=cv2.INTER_LINEAR)

    assert resized.shape[1] == width
    assert resized.shape[2] == 3
    assert resized.shape[0] == 50


def test_resize_maintains_type(sample_image):
    # Test that the output maintains the same dtype as input
    width = 100
    resized = resize(sample_image, width=width)

    assert resized.dtype == sample_image.dtype
