import torch

from torch import nn
from .wdtypes import *


class MultipleLRScheduler(object):

	def __init__(self,schedulers:Dict[str,LRScheduler]):

		instantiated_schedulers = {}
		for model_name, scheduler in schedulers.items():
			if isinstance(scheduler, type):
				instantiated_schedulers[model_name] = scheduler()
			else: instantiated_schedulers[model_name] = scheduler
		self._schedulers = instantiated_schedulers

	def apply(self, optimizers:Dict[str, Optimizer]):
		for model_name, optimizer in optimizers.items():
			self._schedulers[model_name] = self._schedulers[model_name](optimizer)

	def step(self, loss=None):
		for _, sc in self._schedulers.items():
			if 'ReduceLROnPlateau' == sc.__class__.__name__: sc.step(loss)
			else: sc.step()


class StepLR:

	def __init__(self, step_size, gamma=0.1, last_epoch=-1):

		self.step_size = step_size
		self.gamma = gamma
		self.last_epoch = last_epoch

	def __call__(self, optimizer:Optimizer):
		self.sch = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.step_size, gamma=self.gamma,
			last_epoch=self.last_epoch)
		return self.sch


class MultiStepLR:

	def __init__(self, milestones, gamma=0.1, last_epoch=-1):

		self.milestones = milestones
		self.gamma = gamma
		self.last_epoch = last_epoch

	def __call__(self, optimizer:Optimizer):
		self.sch = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.milestones, gamma=self.gamma,
			last_epoch=self.last_epoch)
		return self.sch


class ExponentialLR:

	def __init__(self, gamma, last_epoch=-1):

		self.gamma = gamma
		self.last_epoch = last_epoch

	def __call__(self, optimizer:Optimizer):
		self.sch = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.gamma,
			last_epoch=self.last_epoch)
		return self.sch


class ReduceLROnPlateau:

	def __init__(self, mode='min', factor=0.1, patience=10, verbose=False, threshold=0.0001,
		threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08):

		self.mode=mode
		self.factor=factor
		self.patience=patience
		self.verbose=verbose
		self.threshold=threshold
		self.threshold_mode=threshold_mode
		self.cooldown=cooldown
		self.min_lr=min_lr
		self.eps=eps

	def __call__(self, optimizer:Optimizer):
		self.sch = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=self.mode, factor=self.factor,
					patience=self.patience, verbose=self.verbose, threshold=self.threshold,
					threshold_mode=self.threshold, cooldown=self.cooldown, min_lr=self.min_lr,
					eps=self.eps)
		return self.sch






