import torch

from torch import nn
from fnmatch import fnmatch

from .wdtypes import *

class Initializer(object):

    def __call__(self, model:nn.Module):
        raise NotImplementedError('Initializer must implement this method')


class MultipleInitializers(object):

	def __init__(self, initializers:Dict[str, Initializer]):
	    self._initializers = initializers

	def apply(self, model:nn.Module):
		children = list(model.children())
		for child in children:
			model_name = child.__class__.__name__.lower()
			try:
				child.apply(self._initializers[model_name])
			except KeyError:
				raise Exception('Model name has to be one of: {}'.format(str([child.__class__.__name__.lower() for child in children])))


class Normal(Initializer):

	def __init__(self, mean=0.0, std=0.02, bias=False, pattern='*', pattern_in=True):
	    self.mean = mean
	    self.std = std
	    self.bias = bias
	    self.pattern = pattern
	    super(Normal, self).__init__()

	def __call__(self, submodel:nn.Module):
		for n,p in submodel.named_parameters():
			if fnmatch(n, self.pattern) and pattern_in:
				if self.bias and ('bias' in n):
					nn.init.normal_(p, mean=self.mean, std=self.std)
				elif ('bias' in n):
					continue
				else:
					p.requires_grad: nn.init.normal_(p, mean=self.mean, std=self.std)


class Uniform(Initializer):

	def __init__(self, a=0, b=1, bias=False, pattern='*', pattern_in=True):
	    self.a = a
	    self.b = b
	    self.bias = bias
	    self.pattern = pattern
	    super(Uniform, self).__init__()

	def __call__(self, submodel:nn.Module):
		for n,p in submodel.named_parameters():
			if fnmatch(n, self.pattern) and pattern_in:
				if self.bias and ('bias' in n):
					nn.init.uniform_(p, a=self.a, b=self.b)
				elif ('bias' in n):
					continue
				else:
					p.requires_grad: nn.init.uniform_(p, a=self.a, b=self.b)


class ConstantInitializer(Initializer):

	def __init__(self, value, bias=False, pattern='*'):

	    self.value = value
	    self.pattern = pattern
	    super(ConstantInitializer, self).__init__()

	def __call__(self, submodel:nn.Module):
		for n,p in submodel.named_parameters():
			if fnmatch(n, self.pattern) and pattern_in:
				if self.bias and ('bias' in n):
					nn.init.constant_(p, val=self.value)
				elif ('bias' in n):
					continue
				else:
					p.requires_grad: nn.init.constant_(p, val=self.value)


class XavierUniform(Initializer):

	def __init__(self, gain=1, pattern='*'):
	    self.gain = gain
	    self.pattern = pattern
	    super(XavierUniform, self).__init__()

	def __call__(self, submodel:nn.Module):
		for n,p in submodel.named_parameters():
			if fnmatch(n, self.pattern) and pattern_in:
				if 'bias' in n: nn.init.constant_(p, val=0)
				elif p.requires_grad: nn.init.xavier_uniform_(p, gain=self.gain)


class XavierNormal(Initializer):

	def __init__(self, gain=1, pattern='*'):
	    self.gain = gain
	    self.pattern = pattern
	    super(XavierNormal, self).__init__()

	def __call__(self, submodel:nn.Module):
		for n,p in submodel.named_parameters():
			if fnmatch(n, self.pattern) and pattern_in:
				if 'bias' in n: nn.init.constant_(p, val=0)
				elif p.requires_grad: nn.init.xavier_normal_(p, gain=self.gain)


class KaimingUniform(Initializer):

	def __init__(self, a=0, mode='fan_in', pattern='*'):
	    self.a = a
	    self.mode = mode
	    self.pattern = pattern
	    super(KaimingUniform, self).__init__()

	def __call__(self, submodel:nn.Module):
		for n,p in submodel.named_parameters():
			if fnmatch(n, self.pattern) and pattern_in:
				if 'bias' in n: nn.init.constant_(p, val=0)
				elif p.requires_grad: nn.init.kaiming_uniform_(p, a=self.a, mode=self.mode)


class KaimingNormal(Initializer):

	def __init__(self, a=0, mode='fan_in', pattern='*'):
	    self.a = a
	    self.mode = mode
	    self.pattern = pattern
	    super(KaimingNormal, self).__init__()

	def __call__(self, submodel:nn.Module):
		for n,p in submodel.named_parameters():
			if fnmatch(n, self.pattern) and pattern_in:
				if 'bias' in n: nn.init.constant_(p, val=0)
				elif p.requires_grad: nn.init.kaiming_normal_(p, a=self.a, mode=self.mode)


class Orthogonal(Initializer):

	def __init__(self, gain=1, pattern='*'):
	    self.gain = gain
	    self.pattern = pattern
	    super(Orthogonal, self).__init__()

	def __call__(self, submodel:nn.Module):
		for n,p in submodel.named_parameters():
			if fnmatch(n, self.pattern) and pattern_in:
				if 'bias' in n: nn.init.normal_(p, val=0)
				elif p.requires_grad: nn.init.orthogonal_(p, gain=self.gain)

