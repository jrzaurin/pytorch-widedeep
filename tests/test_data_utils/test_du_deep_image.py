import os

import numpy as np
import pandas as pd

from pytorch_widedeep.preprocessing import ImagePreprocessor

full_path = os.path.realpath(__file__)
path = os.path.split(full_path)[0]
df = pd.DataFrame({"galaxies": ["galaxy1.png", "galaxy2.png"]})
img_col = "galaxies"
imd_dir = os.path.join(path, "images")
processor = ImagePreprocessor(img_col=img_col, img_path=imd_dir)
X_imgs = processor.fit_transform(df)


###############################################################################
# There is not much to test here, since I only resize.
###############################################################################
def test_sizes():
    img_width = X_imgs.shape[1]
    img_height = X_imgs.shape[2]
    assert np.all((img_width == processor.width, img_height == processor.height))
