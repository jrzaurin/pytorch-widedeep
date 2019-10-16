import numpy as np
import string
import torch
import pytest

from torch import nn
from pytorch_widedeep.models.wide import Wide
from pytorch_widedeep.models.deep_dense import DeepDense
from pytorch_widedeep.models.deep_text import DeepText
from pytorch_widedeep.models.deep_image import DeepImage

from pytorch_widedeep.models.wide_deep import WideDeep
from pytorch_widedeep.initializers import (Normal, Uniform,
	ConstantInitializer, XavierNormal, XavierUniform,
	KaimingNormal, KaimingUniform)
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

###############################################################################
# Simply Testing that "something happens (i.e. in!=out)" since the stochastic
# Nature of the most initializers does not allow for more
###############################################################################
initializers_1 = { 'wide': XavierNormal, 'deepdense':XavierUniform, 'deeptext': KaimingNormal,
	'deepimage':KaimingUniform}
test_layers_1 = ['wide.wlinear.weight', 'deepdense.dense.dense_layer_1.0.weight',
	'deeptext.rnn.weight_hh_l1', 'deepimage.dilinear.0.weight']

def test_initializers_1():

	wide = Wide(100, 1)
	deepdense = DeepDense(hidden_layers=[32,16], dropout=[0.5], deep_column_idx=deep_column_idx,
	    embed_input=embed_input, continuous_cols=colnames[-5:], output_dim=1)
	deeptext = DeepText( vocab_size=vocab_size, embed_dim=32, padding_idx=0)
	deepimage=DeepImage(pretrained=True)
	model = WideDeep(wide=wide, deepdense=deepdense, deeptext=deeptext, deepimage=deepimage)
	cmodel = c(model)

	org_weights = []
	for n,p in cmodel.named_parameters():
		if n in test_layers_1: org_weights.append(p)

	model.compile(method='logistic', verbose=0, initializers=initializers_1)
	init_weights = []
	for n,p in model.named_parameters():
		if n in test_layers_1: init_weights.append(p)

	res = all([torch.all((1-(a==b).int()).bool()) for a,b in zip(org_weights, init_weights)])
	assert res

###############################################################################
# Make Sure that the "patterns" parameter works
###############################################################################
initializers_2 = { 'wide': XavierNormal, 'deepdense':XavierUniform,
	'deeptext': KaimingNormal(pattern=r"^(?!.*word_embed).*$")}

def test_initializers_with_pattern():

	wide = Wide(100, 1)
	deepdense = DeepDense(hidden_layers=[32,16], dropout=[0.5], deep_column_idx=deep_column_idx,
	    embed_input=embed_input, continuous_cols=colnames[-5:], output_dim=1)
	deeptext = DeepText( vocab_size=vocab_size, embed_dim=32, padding_idx=0)
	model = WideDeep(wide=wide, deepdense=deepdense, deeptext=deeptext)
	cmodel = c(model)
	org_word_embed = []
	for n,p in cmodel.named_parameters():
		if 'word_embed' in n: org_word_embed.append(p)
	model.compile(method='logistic', verbose=0, initializers=initializers_2)
	init_word_embed = []
	for n,p in model.named_parameters():
		if 'word_embed' in n:  init_word_embed.append(p)

	assert torch.all(org_word_embed[0] == init_word_embed[0])