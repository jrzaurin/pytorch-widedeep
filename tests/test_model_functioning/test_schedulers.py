import numpy as np
import string
import torch
import pytest

from torch import nn
from pytorch_widedeep.models import Wide, DeepDense, DeepText, DeepImage, WideDeep
from pytorch_widedeep.optimizers import Adam, RAdam, SGD, RMSprop
from pytorch_widedeep.lr_schedulers import (StepLR, MultiStepLR, ExponentialLR,
	ReduceLROnPlateau, CyclicLR, OneCycleLR)
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

# target
target = np.random.choice(2, 100)

###############################################################################
# Test that the Step based and Exponential Schedulers functions as expected.
###############################################################################
def test_step_and_exp_lr_schedulers():

	optimizers = { 'wide': Adam, 'deepdense':RAdam, 'deeptext': SGD}
	lr_schedulers = { 'wide': StepLR(step_size=4), 'deepdense':MultiStepLR(milestones=[2,8]),
		'deeptext': ExponentialLR(gamma=0.5)}

	wide = Wide(100, 1)
	deepdense = DeepDense(hidden_layers=[32,16], dropout=[0.5], deep_column_idx=deep_column_idx,
	    embed_input=embed_input, continuous_cols=colnames[-5:], output_dim=1)
	deeptext = DeepText( vocab_size=vocab_size, embed_dim=32, padding_idx=0)
	model = WideDeep(wide=wide, deepdense=deepdense, deeptext=deeptext)
	model.compile(method='logistic', optimizers=optimizers, lr_schedulers=lr_schedulers,
		verbose=1)
	model.fit(X_wide=X_wide, X_deep=X_deep, X_text=X_text, target=target,
		n_epochs=10)

	out = []
	out.append(
		model.optimizer._optimizers['wide'].param_groups[0]['initial_lr'] * 0.1**2 == \
		model.optimizer._optimizers['wide'].param_groups[0]['lr']
		)
	out.append(
		model.optimizer._optimizers['deepdense'].param_groups[0]['initial_lr'] * 0.1**2 == \
		model.optimizer._optimizers['deepdense'].param_groups[0]['lr']
		)
	out.append(
		model.optimizer._optimizers['deeptext'].param_groups[0]['initial_lr'] * 0.5**10 == \
		model.optimizer._optimizers['deeptext'].param_groups[0]['lr'])

	assert all(out)

###############################################################################
# Test that the Cyclic Schedulers functions as expected. At the time of
# writting there is an issue related to the torch_shm_manager in torch v1.3.0
# for pip + OSX. Therefore, I have not tested OneCycleLR which os only
# available for v1.3.0.
###############################################################################
def test_cyclic_lr_schedulers():

	optimizers = { 'wide': Adam(lr=0.001), 'deepdense':Adam(lr=0.001)}
	lr_schedulers = {
		'wide': CyclicLR(base_lr=0.001, max_lr=0.01, step_size_up=20, cycle_momentum=False),
		'deepdense': CyclicLR(base_lr=0.001, max_lr=0.01, step_size_up=10, cycle_momentum=False)}

	wide = Wide(100, 1)
	deepdense = DeepDense(hidden_layers=[32,16], dropout=[0.5], deep_column_idx=deep_column_idx,
	    embed_input=embed_input, continuous_cols=colnames[-5:], output_dim=1)
	model = WideDeep(wide=wide, deepdense=deepdense)
	model.compile(method='logistic', optimizers=optimizers, lr_schedulers=lr_schedulers,
		verbose=0)
	model.fit(X_wide=X_wide, X_deep=X_deep, X_text=X_text, target=target,
		n_epochs=5)

	out = []
	out.append(
		np.isclose(model.optimizer._optimizers['wide'].param_groups[0]['lr'], 0.01)
		)
	out.append(
		model.optimizer._optimizers['deepdense'].param_groups[0]['initial_lr'] == \
		model.optimizer._optimizers['deepdense'].param_groups[0]['lr']
		)

	assert all(out)
