import numpy as np
import torch
import pytest

from copy import deepcopy
from pytorch_widedeep.metrics import BinaryAccuracy, CategoricalAccuracy


y_true = torch.from_numpy(np.random.choice(2, 100)).float()
y_pred = deepcopy(y_true.view(-1,1)).float()

def test_binary_accuracy():
	metric = BinaryAccuracy()
	acc = metric(y_pred, y_true)
	assert acc==1.

@pytest.mark.parametrize(
    'top_k, expected_acc',
    [
    (1, 0.33),
    (2, 0.66)
    ]
    )
def test_categorical_accuracy(top_k, expected_acc):
	y_true = torch.from_numpy(np.random.choice(3, 100))
	y_pred = torch.from_numpy(np.random.rand(100,3))
	metric = CategoricalAccuracy(top_k=top_k)
	acc = metric(y_pred, y_true)
	assert np.isclose(acc, expected_acc, atol=0.3)