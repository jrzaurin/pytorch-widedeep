import numpy as np
import string
import pytest

from pytorch_widedeep.models import Wide, DeepDense, WideDeep

# Wide array
X_wide = np.random.choice(2, (100, 100), p=[0.8, 0.2])

# Deep Array
colnames = list(string.ascii_lowercase)[:10]
embed_cols = [np.random.choice(np.arange(5), 100) for _ in range(5)]
embed_input = [(u, i, j) for u, i, j in zip(colnames[:5], [5] * 5, [16] * 5)]
cont_cols = [np.random.rand(100) for _ in range(5)]
deep_column_idx = {k: v for v, k in enumerate(colnames)}
X_deep = np.vstack(embed_cols + cont_cols).transpose()

# Target
target_regres = np.random.random(100)
target_binary = np.random.choice(2, 100)
target_multic = np.random.choice(3, 100)

# Test dictionary
X_test = {"X_wide": X_wide, "X_deep": X_deep}


##############################################################################
# Test that the three possible methods (regression, binary and mutliclass)
# work well
##############################################################################
@pytest.mark.parametrize(
    "X_wide, X_deep, target, method, X_wide_test, X_deep_test, X_test, output_dim, probs_dim",
    [
        (X_wide, X_deep, target_regres, "regression", X_wide, X_deep, None, 1, None),
        (X_wide, X_deep, target_binary, "binary", X_wide, X_deep, None, 1, 2),
        (X_wide, X_deep, target_multic, "multiclass", X_wide, X_deep, None, 3, 3),
        (X_wide, X_deep, target_regres, "regression", None, None, X_test, 1, None),
        (X_wide, X_deep, target_binary, "binary", None, None, X_test, 1, 2),
        (X_wide, X_deep, target_multic, "multiclass", None, None, X_test, 3, 3),
    ],
)
def test_fit_methods(
    X_wide,
    X_deep,
    target,
    method,
    X_wide_test,
    X_deep_test,
    X_test,
    output_dim,
    probs_dim,
):
    wide = Wide(100, output_dim)
    deepdense = DeepDense(
        hidden_layers=[32, 16],
        dropout=[0.5, 0.5],
        deep_column_idx=deep_column_idx,
        embed_input=embed_input,
        continuous_cols=colnames[-5:],
    )
    model = WideDeep(wide=wide, deepdense=deepdense, output_dim=output_dim)
    model.compile(method=method, verbose=0)
    model.fit(X_wide=X_wide, X_deep=X_deep, target=target)
    preds = model.predict(X_wide=X_wide, X_deep=X_deep, X_test=X_test)
    if method == "binary":
        pass
    else:
        probs = model.predict_proba(X_wide=X_wide, X_deep=X_deep, X_test=X_test)
    assert preds.shape[0] == 100, probs.shape[1] == probs_dim
