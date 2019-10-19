import numpy as np
import string
import torch
import pytest

from torch import nn
from pytorch_widedeep.models import Wide, DeepDense, DeepText, DeepImage, WideDeep
from pytorch_widedeep.optimizers import Adam, RAdam, SGD, RMSprop
from copy import deepcopy as c

# Wide array
X_wide=np.random.choice(2, (100, 100), p = [0.8, 0.2])

# Deep Array
colnames    = list(string.ascii_lowercase)[:10]
embed_cols  = [np.random.choice(np.arange(5), 100) for _ in range(5)]
embed_input = [(u,i,j) for u,i,j in zip(colnames[:5], [5]*5, [16]*5)]
cont_cols   = [np.random.rand(100) for _ in range(5)]
deep_column_idx={k:v for v,k in enumerate(colnames)}
X_deep = np.vstack(embed_cols+cont_cols).transpose()

# Text Array
padded_sequences = np.random.choice(np.arange(1,100), (100, 48))
vocab_size = 1000
X_text = np.hstack((np.repeat(np.array([[0,0]]), 100, axis=0), padded_sequences))

# Image Array
X_img = np.random.choice(256, (100, 224, 224, 3))

optimizers_1 = { 'wide': Adam, 'deepdense':RAdam, 'deeptext': SGD, 'deepimage':RMSprop}
optimizers_2 = { 'wide': RAdam, 'deepdense':SGD, 'deeptext': RMSprop}

###############################################################################
# Test that the MultipleOptimizer class functions as expected
###############################################################################
@pytest.mark.parametrize("optimizers, expected_opt",
    [
        (optimizers_1, { 'wide': 'Adam', 'deepdense':'RAdam', 'deeptext': 'SGD', 'deepimage':'RMSprop'}),
        (optimizers_2, { 'wide': 'RAdam', 'deepdense':'SGD', 'deeptext': 'RMSprop', 'deepimage': 'Adam'}),
    ],
)
def test_optimizers(optimizers, expected_opt):
	wide = Wide(100, 1)
	deepdense = DeepDense(hidden_layers=[32,16], dropout=[0.5], deep_column_idx=deep_column_idx,
	    embed_input=embed_input, continuous_cols=colnames[-5:], output_dim=1)
	deeptext = DeepText( vocab_size=vocab_size, embed_dim=32, padding_idx=0)
	deepimage=DeepImage(pretrained=True)
	model = WideDeep(wide=wide, deepdense=deepdense, deeptext=deeptext, deepimage=deepimage)
	model.compile(method='logistic', optimizers=optimizers)
	out = []
	for name, opt in model.optimizer._optimizers.items():
		out.append(expected_opt[name] == opt.__class__.__name__	)
	assert all(out)

