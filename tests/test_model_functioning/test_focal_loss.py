import string

import numpy as np
import pytest

from pytorch_widedeep.models import Wide, WideDeep, DeepDense

# Wide array
X_wide = np.random.choice(50, (100, 10))

# Deep Array
colnames = list(string.ascii_lowercase)[:10]
embed_cols = [np.random.choice(np.arange(5), 100) for _ in range(5)]
embed_input = [(u, i, j) for u, i, j in zip(colnames[:5], [5] * 5, [16] * 5)]
cont_cols = [np.random.rand(100) for _ in range(5)]
deep_column_idx = {k: v for v, k in enumerate(colnames)}
X_deep = np.vstack(embed_cols + cont_cols).transpose()

# Target
target_binary = np.random.choice(2, 100)
target_multic = np.random.choice(3, 100)


##############################################################################
# Test that the model runs with the focal loss
##############################################################################
@pytest.mark.parametrize(
    "X_wide, X_deep, target, method, pred_dim, probs_dim",
    [
        (X_wide, X_deep, target_binary, "binary", 1, 2),
        (X_wide, X_deep, target_multic, "multiclass", 3, 3),
    ],
)
def test_focal_loss(X_wide, X_deep, target, method, pred_dim, probs_dim):
    wide = Wide(np.unique(X_wide).shape[0], pred_dim)
    deepdense = DeepDense(
        hidden_layers=[32, 16],
        dropout=[0.5, 0.5],
        deep_column_idx=deep_column_idx,
        embed_input=embed_input,
        continuous_cols=colnames[-5:],
    )
    model = WideDeep(wide=wide, deepdense=deepdense, pred_dim=pred_dim)
    model.compile(method=method, verbose=0, with_focal_loss=True)
    model.fit(X_wide=X_wide, X_deep=X_deep, target=target)
    probs = model.predict_proba(X_wide=X_wide, X_deep=X_deep)
    assert probs.shape[1] == probs_dim
