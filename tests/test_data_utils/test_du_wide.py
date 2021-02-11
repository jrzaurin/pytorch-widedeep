import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from pytorch_widedeep.preprocessing import WidePreprocessor


def create_test_dataset(input_type, with_crossed=True):
    df = pd.DataFrame()
    col1 = list(np.random.choice(input_type, 3))
    col2 = list(np.random.choice(input_type, 3))
    df["col1"], df["col2"] = col1, col2
    if with_crossed:
        crossed = ["_".join([str(c1), str(c2)]) for c1, c2 in zip(col1, col2)]
        nuniques = df.col1.nunique() + df.col2.nunique() + len(np.unique(crossed))
    else:
        nuniques = df.col1.nunique() + df.col2.nunique()
    return df, nuniques


some_letters = ["a", "b", "c", "d", "e"]
some_numbers = [1, 2, 3, 4, 5]

wide_cols = ["col1", "col2"]
cross_cols = [("col1", "col2")]


###############################################################################
# Simple test of functionality making sure the shape match
###############################################################################
df_letters, unique_letters = create_test_dataset(some_letters)
df_numbers, unique_numbers = create_test_dataset(some_numbers)
preprocessor1 = WidePreprocessor(wide_cols, cross_cols)


@pytest.mark.parametrize(
    "input_df, expected_shape",
    [(df_letters, unique_letters), (df_numbers, unique_numbers)],
)
def test_preprocessor1(input_df, expected_shape):
    wide_mtx = preprocessor1.fit_transform(input_df)
    assert np.unique(wide_mtx).shape[0] == expected_shape


###############################################################################
# Same test as before but checking that all works when no passing crossed cols
###############################################################################
df_letters_wo_crossed, unique_letters_wo_crossed = create_test_dataset(
    some_letters, with_crossed=False
)
df_numbers_wo_crossed, unique_numbers_wo_crossed = create_test_dataset(
    some_numbers, with_crossed=False
)
preprocessor2 = WidePreprocessor(wide_cols)


@pytest.mark.parametrize(
    "input_df, expected_shape",
    [
        (df_letters_wo_crossed, unique_letters_wo_crossed),
        (df_numbers_wo_crossed, unique_numbers_wo_crossed),
    ],
)
def test_prepare_wide_wo_crossed(input_df, expected_shape):
    wide_mtx = preprocessor2.fit_transform(input_df)
    assert np.unique(wide_mtx).shape[0] == expected_shape


###############################################################################
# test that the inverse transform returns the original DataFrame
###############################################################################
@pytest.mark.parametrize(
    "input_df",
    [
        (df_letters),
        (df_numbers),
    ],
)
def test_inverse_transform(input_df):
    wide_mtx = preprocessor1.fit_transform(input_df)
    org_df = preprocessor1.inverse_transform(wide_mtx)
    org_df = org_df[input_df.columns.tolist()]
    for c in org_df.columns:
        org_df[c] = org_df[c].astype(input_df[c].dtype)
    assert input_df.equals(org_df)


###############################################################################
# Test NotFittedError
###############################################################################


def test_notfittederror():
    processor = WidePreprocessor(wide_cols, cross_cols)
    with pytest.raises(NotFittedError):
        processor.transform(df_letters)
