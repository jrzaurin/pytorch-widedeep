import pandas as pd
from sklearn.exceptions import NotFittedError

from pytorch_widedeep.wdtypes import *  # noqa: F403


# This class does not represent any sctructural advantage, but I keep it to
# keep things tidy, as guidance for contribution and because is useful for the
# check_is_fitted function
class BasePreprocessor:
    """Base Class of All Preprocessors."""

    def __init__(self, *args):
        pass

    def fit(self, df: pd.DataFrame):
        raise NotImplementedError("Preprocessor must implement this method")

    def transform(self, df: pd.DataFrame):
        raise NotImplementedError("Preprocessor must implement this method")

    def fit_transform(self, df: pd.DataFrame):
        raise NotImplementedError("Preprocessor must implement this method")


def check_is_fitted(
    estimator: BasePreprocessor,
    attributes: List[str] = None,
    all_or_any: str = "all",
    condition: bool = True,
):
    r"""Checks if an estimator is fitted

    Parameters
    ----------
    estimator: ``BasePreprocessor``,
        An object of type ``BasePreprocessor``
    attributes: List, default = None
        List of strings with the attributes to check for
    all_or_any: str, default = "all"
        whether all or any of the attributes in the list must be present
    condition: bool, default = True,
        If not attribute list is passed, this condition that must be True for
        the estimator to be considered as fitted
    """

    estimator_name: str = estimator.__class__.__name__
    error_msg = (
        "This {} instance is not fitted yet. Call 'fit' with appropriate "
        "arguments before using this estimator.".format(estimator_name)
    )
    if attributes is not None and all_or_any == "all":
        if not all([hasattr(estimator, attr) for attr in attributes]):
            raise NotFittedError(error_msg)
    elif attributes is not None and all_or_any == "any":
        if not any([hasattr(estimator, attr) for attr in attributes]):
            raise NotFittedError(error_msg)
    elif not condition:
        raise NotFittedError(error_msg)
