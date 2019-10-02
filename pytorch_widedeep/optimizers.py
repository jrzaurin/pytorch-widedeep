import torch

from torch import nn
from .radam import RAdam as orgRAdam
from .wdtypes import *


class MultipleOptimizers(object):

	def __init__(self, optimizers:Dict[str,Optimizer]):

		instantiated_optimizers = {}
		for model_name, optimizer in optimizers.items():
			if isinstance(optimizer, type):
				instantiated_optimizers[model_name] = optimizer()
			else: instantiated_optimizers[model_name] = optimizer
		self._optimizers = instantiated_optimizers

	def apply(self, model:TorchModel):
		children = list(model.children())
		for child in children:
			model_name = child.__class__.__name__.lower()
			self._optimizers[model_name] = self._optimizers[model_name](child)

	def zero_grad(self):
	    for _, opt in self._optimizers.items():
	        opt.zero_grad()

	def step(self):
	    for _, opt in self._optimizers.items():
	        opt.step()

class Adam:

	def __init__(self, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0,
		amsgrad=False):

		self.lr=lr
		self.betas=betas
		self.eps=eps
		self.weight_decay=weight_decay
		self.amsgrad=amsgrad

	def __call__(self, submodel:TorchModel) -> Optimizer:
		self.opt = torch.optim.Adam(submodel.parameters(), lr=self.lr, betas=self.betas, eps=self.eps,
			weight_decay=self.weight_decay, amsgrad=self.amsgrad)
		return self.opt

	def zero_grad(self):
		self.opt.zero_grad()

	def step(self):
		self.opt.step()

class RAdam:

	def __init__(self, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):

		self.lr=lr
		self.betas=betas
		self.eps=eps
		self.weight_decay=weight_decay

	def __call__(self, submodel:TorchModel) -> Optimizer:
		self.opt = orgRAdam(submodel.parameters(), lr=self.lr, betas=self.betas, eps=self.eps,
			weight_decay=self.weight_decay)
		return self.opt

	def zero_grad(self):
		self.opt.zero_grad()

	def step(self):
		self.opt.step()


class SGD:

	def __init__(self, lr=1e-3, momentum=0, dampening=0, weight_decay=0, nesterov=False):

		self.lr=lr
		self.momentum=momentum
		self.dampening=dampening
		self.weight_decay=weight_decay
		self.nesterov=nesterov

	def __call__(self, submodel:TorchModel) -> Optimizer:
		self.opt = torch.optim.SGD(submodel.parameters(), lr=self.lr, momentum=self.momentum,
			dampening=self.dampening, weight_decay=self.weight_decay, nesterov=self.nesterov)
		return self.opt

	def zero_grad(self):
		self.opt.zero_grad()

	def step(self):
		self.opt.step()