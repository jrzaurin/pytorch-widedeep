import numpy as np
import pandas as pd
import pytest

from pytorch_widedeep.datasets import load_adult, load_bio_kdd04


@pytest.mark.parametrize(
    "as_frame",
    [
        (True),
        (False),
    ],
)
def test_load_bio_kdd04(as_frame):
    df = load_bio_kdd04(as_frame=as_frame)
    if as_frame:
        assert (df.shape, type(df)) == ((145751, 77), pd.DataFrame)
    else:
        assert (df.shape, type(df)) == ((145751, 77), np.ndarray)


@pytest.mark.parametrize(
    "as_frame",
    [
        (True),
        (False),
    ],
)
def test_load_adult(as_frame):
    df = load_adult(as_frame=as_frame)
    if as_frame:
        assert (df.shape, type(df)) == ((48842, 15), pd.DataFrame)
    else:
        assert (df.shape, type(df)) == ((48842, 15), np.ndarray)
