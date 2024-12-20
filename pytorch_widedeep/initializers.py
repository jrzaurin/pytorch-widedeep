import re
import warnings
from typing import Any, Dict, Union

from torch import nn

warnings.filterwarnings("default")


class Initializer(object):
    def __call__(self, submodel: nn.Module):
        raise NotImplementedError("Initializer must implement this method")


class MultipleInitializer(object):
    def __init__(
        self, initializers: Dict[str, Union[Initializer, object]], verbose=True
    ):
        self.verbose = verbose
        instantiated_initializers: Dict[str, Any] = {}
        for model_name, initializer in initializers.items():
            if isinstance(initializer, list):
                instantiated_initializers[model_name] = []
                for i in initializer:
                    instantiated_initializers[model_name].append(
                        self._instansiate_maybe(i)
                    )
            else:
                instantiated_initializers[model_name] = self._instansiate_maybe(
                    initializer
                )
        self._initializers = instantiated_initializers

    def apply(self, submodel: Union[nn.Module, nn.ModuleList]):
        for name, child in submodel.named_children():
            try:
                if isinstance(self._initializers[name], list):
                    assert isinstance(child, nn.ModuleList)
                    for i, initializer in enumerate(self._initializers[name]):
                        initializer(child[i])
                else:
                    self._initializers[name](child)
            except KeyError:
                if self.verbose:
                    warnings.warn(
                        "No initializer found for {}".format(name), UserWarning
                    )

    def _instansiate_maybe(self, initializer: object):
        # you passed an object, maybe not an instance of an object
        # and this is crazy, so you know, instantiate it maybe?
        if isinstance(initializer, type):
            return initializer()
        else:
            return initializer


class Normal(Initializer):
    def __init__(self, mean=0.0, std=1.0, bias=False, pattern="."):
        self.mean = mean
        self.std = std
        self.bias = bias
        self.pattern = pattern
        super(Normal, self).__init__()

    def __call__(self, submodel: nn.Module):
        for n, p in submodel.named_parameters():
            if re.search(self.pattern, n):
                if self.bias and ("bias" in n):
                    nn.init.normal_(p, mean=self.mean, std=self.std)
                elif "bias" in n:
                    pass
                elif p.requires_grad:
                    nn.init.normal_(p, mean=self.mean, std=self.std)


class Uniform(Initializer):
    def __init__(self, a=0, b=1, bias=False, pattern="."):
        self.a = a
        self.b = b
        self.bias = bias
        self.pattern = pattern
        super(Uniform, self).__init__()

    def __call__(self, submodel: nn.Module):
        for n, p in submodel.named_parameters():
            if re.search(self.pattern, n):
                if self.bias and ("bias" in n):
                    nn.init.uniform_(p, a=self.a, b=self.b)
                elif "bias" in n:
                    pass
                elif p.requires_grad:
                    nn.init.uniform_(p, a=self.a, b=self.b)


class ConstantInitializer(Initializer):
    def __init__(self, value, bias=False, pattern="."):
        self.bias = bias
        self.value = value
        self.pattern = pattern
        super(ConstantInitializer, self).__init__()

    def __call__(self, submodel: nn.Module):
        for n, p in submodel.named_parameters():
            if re.search(self.pattern, n):
                if self.bias and ("bias" in n):
                    nn.init.constant_(p, val=self.value)
                elif "bias" in n:
                    pass
                elif p.requires_grad:
                    nn.init.constant_(p, val=self.value)


class XavierUniform(Initializer):
    def __init__(self, gain=1, pattern="."):
        self.gain = gain
        self.pattern = pattern
        super(XavierUniform, self).__init__()

    def __call__(self, submodel: nn.Module):
        for n, p in submodel.named_parameters():
            if re.search(self.pattern, n):
                if "bias" in n:
                    nn.init.constant_(p, val=0)
                elif p.requires_grad:
                    try:
                        nn.init.xavier_uniform_(p, gain=self.gain)
                    except Exception:
                        pass


class XavierNormal(Initializer):
    def __init__(self, gain=1, pattern="."):
        self.gain = gain
        self.pattern = pattern
        super(XavierNormal, self).__init__()

    def __call__(self, submodel: nn.Module):
        for n, p in submodel.named_parameters():
            if re.search(self.pattern, n):
                if "bias" in n:
                    nn.init.constant_(p, val=0)
                elif p.requires_grad:
                    try:
                        nn.init.xavier_normal_(p, gain=self.gain)
                    except Exception:
                        pass


class KaimingUniform(Initializer):
    def __init__(self, a=0, mode="fan_in", nonlinearity="leaky_relu", pattern="."):
        self.a = a
        self.mode = mode
        self.nonlinearity = nonlinearity
        self.pattern = pattern
        super(KaimingUniform, self).__init__()

    def __call__(self, submodel: nn.Module):
        for n, p in submodel.named_parameters():
            if re.search(self.pattern, n):
                if "bias" in n:
                    nn.init.constant_(p, val=0)
                elif p.requires_grad:
                    try:
                        nn.init.kaiming_normal_(
                            p, a=self.a, mode=self.mode, nonlinearity=self.nonlinearity
                        )
                    except Exception:
                        pass


class KaimingNormal(Initializer):
    def __init__(self, a=0, mode="fan_in", nonlinearity="leaky_relu", pattern="."):
        self.a = a
        self.mode = mode
        self.nonlinearity = nonlinearity
        self.pattern = pattern
        super(KaimingNormal, self).__init__()

    def __call__(self, submodel: nn.Module):
        for n, p in submodel.named_parameters():
            if re.search(self.pattern, n):
                if "bias" in n:
                    nn.init.constant_(p, val=0)
                elif p.requires_grad:
                    try:
                        nn.init.kaiming_normal_(
                            p, a=self.a, mode=self.mode, nonlinearity=self.nonlinearity
                        )
                    except Exception:
                        pass


class Orthogonal(Initializer):
    def __init__(self, gain=1, pattern="."):
        self.gain = gain
        self.pattern = pattern
        super(Orthogonal, self).__init__()

    def __call__(self, submodel: nn.Module):
        for n, p in submodel.named_parameters():
            if re.search(self.pattern, n):
                if "bias" in n:
                    nn.init.constant_(p, val=0)
                elif p.requires_grad:
                    try:
                        nn.init.orthogonal_(p, gain=self.gain)
                    except Exception:
                        pass
