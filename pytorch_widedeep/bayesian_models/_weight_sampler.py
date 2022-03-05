"""
The code here is greatly insipired by the code at the Blitz package:

https://github.com/piEsposito/blitz-bayesian-deep-learning
"""

import math

from pytorch_widedeep.wdtypes import *  # noqa: F403


class ScaleMixtureGaussianPrior(object):
    r"""Defines the Scale Mixture Prior as proposed in Weight Uncertainty in
    Neural Networks (Eq 7 in the original publication)
    """

    def __init__(self, pi: float, sigma1: float, sigma2: float):
        super().__init__()
        self.pi = pi
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.gaussian1 = torch.distributions.Normal(0, sigma1)
        self.gaussian2 = torch.distributions.Normal(0, sigma2)

    def log_prior(self, input: Tensor) -> Tensor:
        prob1 = torch.exp(self.gaussian1.log_prob(input))
        prob2 = torch.exp(self.gaussian2.log_prob(input))
        return (torch.log(self.pi * prob1 + (1 - self.pi) * prob2)).sum()


class GaussianPosterior(object):
    r"""Defines the Gaussian variational posterior as proposed in Weight
    Uncertainty in Neural Networks
    """

    def __init__(self, param_mu: Tensor, param_rho: Tensor):
        super().__init__()
        self.param_mu = param_mu
        self.param_rho = param_rho
        self.normal = torch.distributions.Normal(0, 1)

    @property
    def sigma(self):
        return torch.log1p(torch.exp(self.param_rho))

    def sample(self) -> Tensor:
        epsilon = self.normal.sample(self.param_rho.size()).to(self.param_rho.device)
        return self.param_mu + self.sigma * epsilon

    def log_posterior(self, input: Tensor) -> Tensor:
        return (
            -math.log(math.sqrt(2 * math.pi))
            - torch.log(self.sigma)
            - ((input - self.param_mu) ** 2) / (2 * self.sigma**2)
        ).sum()
