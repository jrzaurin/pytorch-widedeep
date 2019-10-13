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


class CyclicLR:

	def __init__(self, base_lr, max_lr, step_size_up=2000, step_size_down=None,
		mode='triangular', gamma=1.0, scale_fn=None, scale_mode='cycle',
		cycle_momentum=True, base_momentum=0.8, max_momentum=0.9, last_epoch=-1):

		self.base_lr = base_lr
		self.max_lr = max_lr
		self.step_size_up = step_size_up
		self.step_size_down = step_size_down
		self.mode = mode
		self.gamma = gamma
		self.scale_fn = scale_fn
		self.scale_mode = scale_mode
		self.cycle_momentum = cycle_momentum
		self.base_momentum = base_momentum
		self.max_momentum = max_momentum
		self.last_epoch = last_epoch

	def __call__(self, optimizer:Optimizer):
		self.sch = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=self.base_lr,
			max_lr=self.max_lr, step_size_up=self.step_size_up, step_size_down=self.step_size_down,
			mode=self.mode, gamma=self.gamma, scale_fn=self.scale_fn, scale_mode=self.scale_mode,
			cycle_momentum=self.cycle_momentum, base_momentum=self.base_momentum,
			max_momentum=self.max_momentum, last_epoch=self.last_epoch)
		return self.sch


class OneCycleLR:

	def __init__(self, max_lr, total_steps=None, epochs=None, steps_per_epoch=None,
		pct_start=0.3, anneal_strategy='cos', cycle_momentum=True, base_momentum=0.85,
		max_momentum=0.95, div_factor=25.0, final_div_factor=10000.0, last_epoch=-1):

		self.max_lr = max_lr
		self.total_steps = total_steps
		self.epochs = epochs
		self.steps_per_epoch = steps_per_epoch
		self.pct_start = pct_start
		self.anneal_strategy = anneal_strategy
		self.cycle_momentum = cycle_momentum
		self.base_momentum = base_momentum
		self.max_momentum = max_momentum
		self.div_factor = div_factor
		self.final_div_factor = final_div_factor
		self.last_epoch = last_epoch

	def __call__(self, optimizer:Optimizer):
		self.sch = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr = self.max_lr,
			total_steps = self.total_steps, epochs = self.epochs,
			steps_per_epoch = self.steps_per_epoch, pct_start = self.pct_start,
			anneal_strategy = self.anneal_strategy, cycle_momentum = self.cycle_momentum,
			base_momentum = self.base_momentum, max_momentum = self.max_momentum,
			div_factor = self.div_factor, final_div_factor = self.final_div_factor,
			last_epoch = self.last_epoch)
		return self.sch
