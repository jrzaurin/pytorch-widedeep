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
    def __init__(
        self,
        in_features: int,
        out_features: int,
        use_bias: bool = True,
        prior_sigma_1: float = 0.1,
        prior_sigma_2: float = 0.002,
        prior_pi: float = 1.0,
        posterior_mu_init: float = 0.0,
        posterior_rho_init: float = -6.0,
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

        # Variational weight and bias parameters and sample for the posterior
        # computation
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
        if self.prior_sigma_1 != 0.1:
            s + ", prior_sigma_1={prior_sigma_1}"
        if self.prior_sigma_2 != 0.002:
            s + ", prior_sigma_2={prior_sigma_2}"
        if self.prior_pi != 1.0:
            s + ", prior_pi={prior_pi}"
        if self.posterior_mu_init != 0.0:
            s + ", posterior_mu_init={posterior_mu_init}"
        if self.posterior_rho_init != -6.0:
            s + ", posterior_rho_init={posterior_rho_init}"
        return s.format(**self.__dict__)
