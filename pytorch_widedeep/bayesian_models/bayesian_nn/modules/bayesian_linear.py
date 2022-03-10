"""
The code here is greatly insipired by a couple of sources:

the Blitz package: https://github.com/piEsposito/blitz-bayesian-deep-learning and

Weight Uncertainty in Neural Networks post by Nitarshan Rajkumar: https://www.nitarshan.com/bayes-by-backprop/

and references therein
"""

import torch.nn.functional as F
from torch import nn

from pytorch_widedeep.wdtypes import *  # noqa: F403
from pytorch_widedeep.bayesian_models._weight_sampler import (
    GaussianPosterior,
    ScaleMixtureGaussianPrior,
)
from pytorch_widedeep.bayesian_models._base_bayesian_model import (
    BayesianModule,
)


class BayesianLinear(BayesianModule):
    r"""Applies a linear transformation to the incoming data as proposed in Weight
    Uncertainity on Neural Networks

    Parameters
    ----------
    in_features: int
        size of each input sample
    out_features: int
         size of each output sample
    use_bias: bool, default = True
        Boolean indicating if an additive bias will be learnt
    prior_sigma_1: float, default = 1.0
        Prior of the sigma parameter for the first of the two Gaussian
        distributions that will be mixed to produce the prior weight
        distribution
    prior_sigma_2: float, default = 0.002
        Prior of the sigma parameter for the second of the two Gaussian
        distributions that will be mixed to produce the prior weight
        distribution
    prior_pi: float, default = 0.8
        Scaling factor that will be used to mix the Gaussians to produce the
        prior weight distribution
    posterior_mu_init: float = 0.0
        The posterior sample of the weights is defined as:

        .. math::
           \begin{aligned}
           \mathbf{w} &= \mu + log(1 + exp(\rho))
           \end{aligned}

        where:

        .. math::
           \begin{aligned}
           \mathcal{N}(x\vert \mu, \sigma) &= \frac{1}{\sqrt{2\pi}\sigma}e^{-\frac{(x-\mu)^2}{2\sigma^2}}\\
           \log{\mathcal{N}(x\vert \mu, \sigma)} &= -\log{\sqrt{2\pi}} -\log{\sigma} -\frac{(x-\mu)^2}{2\sigma^2}\\
           \end{aligned}

        :math:`\mu` is initialised using a normal distributtion with mean
        ``posterior_rho_init`` and std equal to 0.1.
    posterior_rho_init: float = -7.0
        As in the case of :math:`\mu`, :math:`\rho` is initialised using a
        normal distributtion with mean ``posterior_rho_init`` and std equal to
        0.1.

    Examples
    --------
    >>> import torch
    >>> from pytorch_widedeep.bayesian_models import bayesian_nn as bnn
    >>> linear = bnn.BayesianLinear(10, 6)
    >>> input = torch.rand(6, 10)
    >>> out = linear(input)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        use_bias: bool = True,
        prior_sigma_1: float = 1.0,
        prior_sigma_2: float = 0.002,
        prior_pi: float = 0.8,
        posterior_mu_init: float = 0.0,
        posterior_rho_init: float = -7.0,
    ):
        super(BayesianLinear, self).__init__()

        # main parameters of the layer
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = use_bias

        # posterior params
        self.posterior_mu_init = posterior_mu_init
        self.posterior_rho_init = posterior_rho_init
        self.prior_sigma_1 = prior_sigma_1
        self.prior_sigma_2 = prior_sigma_2
        self.prior_pi = prior_pi

        # Variational Posterior
        self.weight_mu = nn.Parameter(
            torch.Tensor(out_features, in_features).normal_(posterior_mu_init, 0.1)
        )
        self.weight_rho = nn.Parameter(
            torch.Tensor(out_features, in_features).normal_(posterior_rho_init, 0.1)
        )
        self.weight_sampler = GaussianPosterior(self.weight_mu, self.weight_rho)

        if self.use_bias:
            self.bias_mu = nn.Parameter(
                torch.Tensor(out_features).normal_(posterior_mu_init, 0.1)
            )
            self.bias_rho = nn.Parameter(
                torch.Tensor(out_features).normal_(posterior_rho_init, 0.1)
            )
            self.bias_sampler = GaussianPosterior(self.bias_mu, self.bias_rho)
        else:
            self.bias_mu, self.bias_rho = None, None

        # Prior
        self.weight_prior_dist = ScaleMixtureGaussianPrior(
            self.prior_pi,
            self.prior_sigma_1,
            self.prior_sigma_2,
        )
        if self.use_bias:
            self.bias_prior_dist = ScaleMixtureGaussianPrior(
                self.prior_pi,
                self.prior_sigma_1,
                self.prior_sigma_2,
            )

        self.log_prior: Union[Tensor, float] = 0.0
        self.log_variational_posterior: Union[Tensor, float] = 0.0

    def forward(self, X: Tensor) -> Tensor:

        if not self.training:
            return F.linear(X, self.weight_mu, self.bias_mu)

        weight = self.weight_sampler.sample()
        if self.use_bias:
            bias = self.bias_sampler.sample()
            bias_log_posterior: Union[Tensor, float] = self.bias_sampler.log_posterior(
                bias
            )
            bias_log_prior: Union[Tensor, float] = self.bias_prior_dist.log_prior(bias)
        else:
            bias = None
            bias_log_posterior = 0.0
            bias_log_prior = 0.0

        self.log_variational_posterior = (
            self.weight_sampler.log_posterior(weight) + bias_log_posterior
        )
        self.log_prior = self.weight_prior_dist.log_prior(weight) + bias_log_prior

        return F.linear(X, weight, bias)

    def extra_repr(self) -> str:  # noqa: C901
        s = "{in_features}, {out_features}"
        if self.use_bias is not False:
            s += ", use_bias=True"
        if self.prior_sigma_1 != 1.0:
            s += ", prior_sigma_1={prior_sigma_1}"
        if self.prior_sigma_2 != 0.002:
            s += ", prior_sigma_2={prior_sigma_2}"
        if self.prior_pi != 0.8:
            s += ", prior_pi={prior_pi}"
        if self.posterior_mu_init != 0.0:
            s += ", posterior_mu_init={posterior_mu_init}"
        if self.posterior_rho_init != -7.0:
            s += ", posterior_rho_init={posterior_rho_init}"
        return s.format(**self.__dict__)
