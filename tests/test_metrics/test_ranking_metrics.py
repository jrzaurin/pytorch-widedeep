# Thanks claude 3.5 for the test cases.
import numpy as np
import torch
import pytest

from pytorch_widedeep.metrics import (
    MAP_at_k,
    NDCG_at_k,
    Recall_at_k,
    HitRatio_at_k,
    Precision_at_k,
    BinaryNDCG_at_k,
)


@pytest.fixture
def setup_data():
    y_pred = torch.tensor(
        [
            [0.7, 0.9, 0.5, 0.8, 0.6],
            [0.3, 0.1, 0.5, 0.2, 0.4],
        ]
    )
    y_true = torch.tensor(
        [
            [0, 1, 0, 1, 1],
            [1, 1, 0, 0, 0],
        ]
    )
    return y_pred, y_true


@pytest.fixture
def setup_data_ndcg():
    y_pred = torch.tensor(
        [
            [0.7, 0.9, 0.5, 0.8, 0.6],
            [0.3, 0.1, 0.5, 0.2, 0.4],
        ]
    )
    # Using relevance scores from 0 to 3
    y_true = torch.tensor(
        [
            [1, 3, 1, 2, 0],
            [3, 1, 0, 2, 1],
        ]
    )
    return y_pred, y_true


def test_binary_ndcg_at_k(setup_data):
    y_pred, y_true = setup_data
    binary_ndcg = BinaryNDCG_at_k(n_cols=5, k=3)
    result = binary_ndcg(y_pred.flatten(), y_true.flatten())
    expected = np.array(0.5719)
    np.testing.assert_almost_equal(result, expected, decimal=4)


def test_map_at_k(setup_data):
    y_pred, y_true = setup_data
    map_at_k = MAP_at_k(n_cols=5, k=3)
    result = map_at_k(y_pred.flatten(), y_true.flatten())
    expected = np.array(0.4166)
    np.testing.assert_almost_equal(result, expected, decimal=4)


def test_hit_ratio_at_k(setup_data):
    y_pred, y_true = setup_data
    hr_at_k = HitRatio_at_k(n_cols=5, k=3)
    result = hr_at_k(y_pred.flatten(), y_true.flatten())
    expected = np.array(1.0)
    np.testing.assert_almost_equal(result, expected, decimal=4)


def test_precision_at_k(setup_data):
    y_pred, y_true = setup_data
    prec_at_k = Precision_at_k(n_cols=5, k=3)
    result = prec_at_k(y_pred.flatten(), y_true.flatten())
    expected = np.array(0.5)
    np.testing.assert_almost_equal(result, expected, decimal=4)


def test_recall_at_k(setup_data):
    y_pred, y_true = setup_data
    rec_at_k = Recall_at_k(n_cols=5, k=3)
    result = rec_at_k(y_pred.flatten(), y_true.flatten())
    expected = np.array(0.5833)
    np.testing.assert_almost_equal(result, expected, decimal=4)


def test_edge_cases_all_relevant_items():
    # Test with all relevant items
    y_pred = torch.tensor([0.9, 0.8, 0.7, 0.6, 0.5])
    y_true = torch.tensor([1, 1, 1, 1, 1])

    ndcg = NDCG_at_k(n_cols=5, k=3)
    assert ndcg(y_pred, y_true) == 1.0

    binary_ndcg = BinaryNDCG_at_k(n_cols=5, k=3)
    assert binary_ndcg(y_pred, y_true) == 1.0

    map_at_k = MAP_at_k(n_cols=5, k=3)
    assert np.isclose(map_at_k(y_pred, y_true), 0.6)

    hr_at_k = HitRatio_at_k(n_cols=5, k=3)
    assert hr_at_k(y_pred, y_true) == 1.0

    prec_at_k = Precision_at_k(n_cols=5, k=3)
    assert prec_at_k(y_pred, y_true) == 1.0

    rec_at_k = Recall_at_k(n_cols=5, k=3)
    assert np.isclose(rec_at_k(y_pred, y_true), 0.6)


def test_edge_cases_no_relevant_items():

    # Test with no relevant items
    y_pred = torch.tensor([0.9, 0.8, 0.7, 0.6, 0.5])
    y_true = torch.tensor([0, 0, 0, 0, 0])

    ndcg = NDCG_at_k(n_cols=5, k=3)
    assert ndcg(y_pred, y_true) == 0.0

    binary_ndcg = BinaryNDCG_at_k(n_cols=5, k=3)
    assert binary_ndcg(y_pred, y_true) == 0.0

    map_at_k = MAP_at_k(n_cols=5, k=3)
    assert map_at_k(y_pred, y_true) == 0.0

    hr_at_k = HitRatio_at_k(n_cols=5, k=3)
    assert hr_at_k(y_pred, y_true) == 0.0

    prec_at_k = Precision_at_k(n_cols=5, k=3)
    assert prec_at_k(y_pred, y_true) == 0.0

    rec_at_k = Recall_at_k(n_cols=5, k=3)
    assert rec_at_k(y_pred, y_true) == 0.0


def test_k_greater_than_n_cols():
    with pytest.raises(ValueError):
        NDCG_at_k(n_cols=5, k=10)
    with pytest.raises(ValueError):
        BinaryNDCG_at_k(n_cols=5, k=10)
    with pytest.raises(ValueError):
        MAP_at_k(n_cols=5, k=10)
    with pytest.raises(ValueError):
        HitRatio_at_k(n_cols=5, k=10)
    with pytest.raises(ValueError):
        Precision_at_k(n_cols=5, k=10)
    with pytest.raises(ValueError):
        Recall_at_k(n_cols=5, k=10)


def test_ndcg_at_k(setup_data_ndcg):
    y_pred, y_true = setup_data_ndcg
    ndcg = NDCG_at_k(n_cols=5, k=3)
    result = ndcg(y_pred.flatten(), y_true.flatten())
    expected = np.array(0.7198)
    np.testing.assert_almost_equal(result, expected, decimal=4)


def test_ndcg_at_k_edge_cases():
    # Test with non-decreasing ranking
    y_pred = torch.tensor([0.5, 0.6, 0.7, 0.8, 0.9])
    y_true = torch.tensor([0, 1, 2, 3, 4])

    ndcg = NDCG_at_k(n_cols=5, k=3)
    result = ndcg(y_pred, y_true)
    assert result == 1.0

    # Test with non-increasing ranking
    ndcg = NDCG_at_k(n_cols=5, k=3)
    y_pred = torch.tensor([0.9, 0.8, 0.7, 0.6, 0.5])
    result = ndcg(y_pred, y_true)
    expected = np.array(0.1019)
    np.testing.assert_almost_equal(result, expected, decimal=4)

    # Test with all zero relevance
    ndcg = NDCG_at_k(n_cols=5, k=3)
    y_true = torch.tensor([0, 0, 0, 0, 0])
    result = ndcg(y_pred, y_true)
    assert result == 0.0


def test_ndcg_at_k_k_greater_than_n_cols():
    with pytest.raises(ValueError):
        NDCG_at_k(n_cols=5, k=10)
