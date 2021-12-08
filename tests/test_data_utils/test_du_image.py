import os

import cv2
import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from pytorch_widedeep.utils import SimplePreprocessor, AspectAwarePreprocessor
from pytorch_widedeep.preprocessing import ImagePreprocessor

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
# Test NotFittedError
###############################################################################


def test_notfittederror():
    processor = ImagePreprocessor(img_col=img_col, img_path=imd_dir)
    with pytest.raises(NotFittedError):
        processor.transform(df)
