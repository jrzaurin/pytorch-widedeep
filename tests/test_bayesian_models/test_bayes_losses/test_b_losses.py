import numpy as np
import torch
from sklearn.metrics import mean_squared_error

from pytorch_widedeep.losses import BayesianSELoss, BayesianRegressionLoss


##############################################################################
# BayesianSELoss
##############################################################################
def test_mse_based_losses():
    y_true = np.array([3, 5, 2.5, 7]).reshape(-1, 1)
    y_pred = np.array([2.5, 5, 4, 8]).reshape(-1, 1)

    t_true = torch.from_numpy(y_true)
    t_pred = torch.from_numpy(y_pred)

    are_close = np.isclose(
        mean_squared_error(y_true, y_pred),
        BayesianSELoss()(t_pred, t_true).item() * 0.5,
    )

    assert are_close
