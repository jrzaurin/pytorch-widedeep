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

    # Explaining the expected value (@k=3):
    # DCG_row1 = 0*1 + 1*0.6309 + 1*0.5 = 1.1309 -> (0.7 -> 0, 0.9 -> 1, 0.8 -> 1)
    # DCG_row2 = 1*1 + 0*0.6309 + 0*0.5 = 1.0 -> (0.3 -> 1, 0.5 -> 0, 0.4 -> 0)
    # IDCG_row1 = 1*1 + 1*0.6309 + 1*0.5 = 2.1309 (3 relevant items)
    # IDCG_row2 = 1*1 + 1*0.6309 = 1.6309 (2 relevant items)
    # NDCG_row1 = DCG_row1 / IDCG_row1 = 1.1309 / 2.1309 = 0.5309
    # NDCG_row2 = DCG_row2 / IDCG_row2 = 1.0 / 1.6309 = 0.6129
    # binary_NDCG = (NDCG_row1 + NDCG_row2) / 2 = (0.5309 + 0.6129) / 2 = 0.5719


def test_map_at_k(setup_data):
    y_pred, y_true = setup_data
    map_at_k = MAP_at_k(n_cols=5, k=3)
    result = map_at_k(y_pred.flatten(), y_true.flatten())
    expected = np.array(0.4166)
    np.testing.assert_almost_equal(result, expected, decimal=4)

    # Explaining the expected value (@k=3):
    # batch relevance for row top 3 predictions: [[1, 1, 0], [0, 0, 1]]
    # AP_row1 = (1/1 + 2/2 + 0/3) / 3 = 0.6666 -> [(1/1) * 1 + (2/2) * 1 + (2/3) * 0] / 3
    # AP_row2 = (0/1 + 0/2 + 1/3) / 2 = 0.1666 -> [(0/1) * 0 + (0/2) * 0 + (1/3) * 1] / 2
    # MAP = (AP_row1 + AP_row2) / 2 = (0.6666 + 0.1666) / 2 = 0.4166


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

    # Explaining the expected value (@k=3):
    # batch relevance for row top 3 predictions: [[1, 1, 0], [0, 0, 1]]
    # Precision_row1 = 2 / 3 = 0.666
    # Precision_row2 = 1 / 3 = 0.333
    # Precision = (Precision_row1 + Precision_row2) / 2 = (0.666 + 0.333) / 2 = 0.5
    # caveat: if k is higher than the number of relevant items for a given row
    # this metric will never be 1.0


def test_recall_at_k(setup_data):
    y_pred, y_true = setup_data
    rec_at_k = Recall_at_k(n_cols=5, k=3)
    result = rec_at_k(y_pred.flatten(), y_true.flatten())
    expected = np.array(0.5833)
    np.testing.assert_almost_equal(result, expected, decimal=4)

    # Explaining the expected value (@k=3):
    # batch relevance for row top 3 predictions: [[1, 1, 0], [0, 0, 1]]
    # Recall_row1 = 2 / 3 = 0.666
    # Recall_row2 = 1 / 2 = 0.5
    # Recall = (Recall_row1 + Recall_row2) / 2 = (0.666 + 0.5) / 2 = 0.5833


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

    # Explaining the expected value (@k=3):
    # top 3 predictions for row 1: [0.9, 0.8, 0.7] -> [3, 2, 1]
    # top 3 predictions for row 2: [0.5, 0.4, 0.3] -> [0, 1, 3]
    # DCG = sum[(2^rel - 1) / log2(rank + 1)]
    # DCG_row1 = 2^3 - 1 / log2(1+1) + 2^2 - 1 / log2(2+1) + 2^1 - 1 / log2(3+1) = 7.0 + 1.8928 + 0.5000 = 9.3928
    # DCG_row2 = 2^0 - 1 / log2(1+1) + 2^1 - 1 / log2(2+1) + 2^3 - 1 / log2(3+1) = 0.0 + 0.6309 + 3.5 = 4.1309
    # IDCG = sum[(2^rel - 1) / log2(rank + 1)] for the ideal ranking at k=3 [[3, 2, 1], [3, 2, 1]]
    # IDCG_row1 = 2^3 - 1 / log2(1+1) + 2^2 - 1 / log2(2+1) + 2^1 - 1 / log2(3+1) = 7.0 + 1.8928 + 0.5000 = 9.3928
    # IDCG_row2 = IDCG_row1 = 9.3928
    # NDGC_row1 = DCG_row1 / IDCG_row1 = 9.3928 / 9.3928 = 1.0
    # NDGC_row2 = DCG_row2 / IDCG_row2 = 4.1309 / 9.3928 = 0.4398
    # NDCG = (NDGC_row1 + NDGC_row2) / 2 = (1.0 + 0.4398) / 2 = 0.7198


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
