import string

import numpy as np
import pytest
from sklearn.model_selection import train_test_split

from pytorch_widedeep.training import BayesianTrainer
from pytorch_widedeep.bayesian_models import BayesianWide, BayesianTabMlp

np.random.seed(1)

# Wide array
X_wide = np.random.choice(50, (32, 10))

# Tab Array
colnames = list(string.ascii_lowercase)[:10]
embed_cols = [np.random.choice(np.arange(5), 32) for _ in range(5)]
embed_input = [(u, i, j) for u, i, j in zip(colnames[:5], [5] * 5, [16] * 5)]
cont_cols = [np.random.rand(32) for _ in range(5)]
X_tabmlp = np.vstack(embed_cols + cont_cols).transpose()

# Target
target = np.random.choice(2, 32)

# train/validation split
(
    X_wide_tr,
    X_wide_val,
    X_tabmlp_tr,
    X_tabmlp_val,
    y_train,
    y_val,
) = train_test_split(X_wide, X_tabmlp, target)

# build model components
wide = BayesianWide(np.unique(X_wide).shape[0], 1)
tabmlp = BayesianTabMlp(
    column_idx={k: v for v, k in enumerate(colnames)},
    cat_embed_input=embed_input,
    continuous_cols=colnames[-5:],
    mlp_hidden_dims=[32, 16],
)

# #############################################################################
# Test that runs with different data inputs
# #############################################################################


@pytest.mark.parametrize(
    "model, X_tab, target, X_tab_val, target_val , val_split",
    [
        (wide, X_wide, target, None, None, 0.2),
        (wide, X_wide_tr, y_train, X_wide_val, y_val, None),
        (tabmlp, X_tabmlp, target, None, None, 0.2),
        (tabmlp, X_tabmlp_tr, y_train, X_tabmlp_val, y_val, None),
    ],
)
def test_data_input_options(model, X_tab, target, X_tab_val, target_val, val_split):
    trainer = BayesianTrainer(model, objective="binary", verbose=0)

    trainer.fit(
        X_tab=X_tab,
        target=target,
        X_tab_val=X_tab_val,
        target_val=target_val,
        val_split=val_split,
        batch_size=16,
    )

    assert trainer.history["train_loss"] is not None
