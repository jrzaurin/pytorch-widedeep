import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from pytorch_widedeep.preprocessing.base_preprocessor import (
    BasePreprocessor,
    check_is_fitted,
)

df = pd.DataFrame({"col1": ["a", "b", "c", "d", "e"], "col2": [1, 2, 3, 4, 5]})


class DummyPreprocessor(BasePreprocessor):
    def __init__(self):
        super().__init__()

    def fit(self, df):
        self.att1 = 1
        self.att2 = 2
        return df

    def transform(self, df):
        check_is_fitted(self, attributes=["att1", "att2"], all_or_any="any")
        return df

    def fit_transform(self, df):
        return self.fit(df).transform(df)


class IncompletePreprocessor(BasePreprocessor):
    def __init__(self):
        super().__init__()

    def fit(self, df):
        return df

    def transform(self, df):
        return df


###############################################################################
# test check_is_fitted with "any"
###############################################################################


def test_check_is_fitted():
    dummy_preprocessor = DummyPreprocessor()
    with pytest.raises(NotFittedError):
        dummy_preprocessor.transform(df)


###############################################################################
# test base_preprocessor raising NotImplemented error
###############################################################################


def test_base_non_implemented_error():
    with pytest.raises(NotImplementedError):
        incomplete_preprocessor = IncompletePreprocessor()  # noqa: F841
        incomplete_preprocessor.fit_transform(df)
