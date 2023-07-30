import numpy as np
import pandas as pd
import pytest

from pytorch_widedeep.datasets import (
    load_rf1,
    load_adult,
    load_birds,
    load_ecoli,
    load_bio_kdd04,
    load_womens_ecommerce,
    load_california_housing,
    load_movielens100k,
)


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


@pytest.mark.parametrize(
    "as_frame",
    [
        (True),
        (False),
    ],
)
def test_load_ecoli(as_frame):
    df = load_ecoli(as_frame=as_frame)
    if as_frame:
        assert (df.shape, type(df)) == ((336, 9), pd.DataFrame)
    else:
        assert (df.shape, type(df)) == ((336, 9), np.ndarray)


@pytest.mark.parametrize(
    "as_frame",
    [
        (True),
        (False),
    ],
)
def test_load_womens_ecommerce(as_frame):
    df = load_womens_ecommerce(as_frame=as_frame)
    if as_frame:
        assert (df.shape, type(df)) == ((23486, 10), pd.DataFrame)
    else:
        assert (df.shape, type(df)) == ((23486, 10), np.ndarray)


@pytest.mark.parametrize(
    "as_frame",
    [
        (True),
        (False),
    ],
)
def test_load_rf1(as_frame):
    df = load_rf1(as_frame=as_frame)
    if as_frame:
        assert (df.shape, type(df)) == ((4108, 72), pd.DataFrame)
    else:
        assert (df.shape, type(df)) == ((4108, 72), np.ndarray)


@pytest.mark.parametrize(
    "as_frame",
    [
        (True),
        (False),
    ],
)
def test_load_birds(as_frame):
    df = load_birds(as_frame=as_frame)
    if as_frame:
        assert (df.shape, type(df)) == ((322, 279), pd.DataFrame)
    else:
        assert (df.shape, type(df)) == ((322, 279), np.ndarray)


@pytest.mark.parametrize(
    "as_frame",
    [
        (True),
        (False),
    ],
)
def test_load_california_housing(as_frame):
    df = load_california_housing(as_frame=as_frame)
    if as_frame:
        assert (df.shape, type(df)) == ((20640, 9), pd.DataFrame)
    else:
        assert (df.shape, type(df)) == ((20640, 9), np.ndarray)


@pytest.mark.parametrize(
    "as_frame",
    [
        (True),
        (False),
    ],
)
def test_load_movielens100k(as_frame):
    df_data, df_items, df_users = load_movielens100k(as_frame=as_frame)
    if as_frame:
        assert (
            df_data.shape,
            df_items.shape,
            df_users.shape,
            type(df_data),
            type(df_items),
            type(df_users),
        ) == (
            (100000, 4),
            (1682, 24),
            (943, 5),
            pd.DataFrame,
            pd.DataFrame,
            pd.DataFrame,
        )
    else:
        assert (
            df_data.shape,
            df_items.shape,
            df_users.shape,
            type(df_data),
            type(df_items),
            type(df_users),
        ) == (
            (100000, 4),
            (1682, 24),
            (943, 5),
            np.ndarray,
            np.ndarray,
            np.ndarray,
        )
