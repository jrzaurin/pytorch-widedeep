import numpy as np
import torch
from torch import nn

from pytorch_widedeep.wdtypes import Dict, List, Tuple, Union, Tensor, Optional
from pytorch_widedeep.bayesian_models import bayesian_nn as bnn
from pytorch_widedeep.models._get_activation_fn import get_activation_fn
from pytorch_widedeep.bayesian_models._weight_sampler import (
    GaussianPosterior,
    ScaleMixtureGaussianPrior,
)
from pytorch_widedeep.bayesian_models._base_bayesian_model import (
    BayesianModule,
)

NormLayers = Union[nn.Identity, nn.LayerNorm, nn.BatchNorm1d]


class BayesianContEmbeddings(BayesianModule):
    def __init__(
        self,
        n_cont_cols: int,
        embed_dim: int,
        prior_sigma_1: float,
        prior_sigma_2: float,
        prior_pi: float,
        posterior_mu_init: float,
        posterior_rho_init: float,
        use_bias: bool = False,
        activation_fn: Optional[str] = None,
    ):
        super(BayesianContEmbeddings, self).__init__()

        self.n_cont_cols = n_cont_cols
        self.embed_dim = embed_dim
        self.use_bias = use_bias

        self.weight_mu = nn.Parameter(
            torch.Tensor(n_cont_cols, embed_dim).normal_(posterior_mu_init, 0.1)
        )
        self.weight_rho = nn.Parameter(
            torch.Tensor(n_cont_cols, embed_dim).normal_(posterior_rho_init, 0.1)
        )
        self.weight_sampler = GaussianPosterior(self.weight_mu, self.weight_rho)

        if use_bias:
            self.bias_mu = nn.Parameter(
                torch.Tensor(n_cont_cols, embed_dim).normal_(posterior_mu_init, 0.1)
            )
            self.bias_rho = nn.Parameter(
                torch.Tensor(n_cont_cols, embed_dim).normal_(posterior_rho_init, 0.1)
            )
            self.bias_sampler = GaussianPosterior(self.bias_mu, self.bias_rho)
        else:
            self.bias_mu, self.bias_rho = None, None

        # Prior
        self.weight_prior_dist = ScaleMixtureGaussianPrior(
            prior_pi,
            prior_sigma_1,
            prior_sigma_2,
        )
        if self.use_bias:
            self.bias_prior_dist = ScaleMixtureGaussianPrior(
                prior_pi,
                prior_sigma_1,
                prior_sigma_2,
            )

        self.log_prior: Union[Tensor, float] = 0.0
        self.log_variational_posterior: Union[Tensor, float] = 0.0

        self.activation_fn = (
            get_activation_fn(activation_fn) if activation_fn is not None else None
        )

    def forward(self, X: Tensor) -> Tensor:
        if not self.training:
            x = self.weight_mu.unsqueeze(0) * X.unsqueeze(2)
            if self.bias_mu is not None:
                x + self.bias_mu.unsqueeze(0)
            return x

        weight = self.weight_sampler.sample()
        if self.use_bias:
            bias = self.bias_sampler.sample()
            bias_log_posterior: Union[Tensor, float] = self.bias_sampler.log_posterior(
                bias
            )
            bias_log_prior: Union[Tensor, float] = self.bias_prior_dist.log_prior(bias)
        else:
            bias = 0.0  # type: ignore[assignment]
            bias_log_posterior = 0.0
            bias_log_prior = 0.0

        self.log_variational_posterior = (
            self.weight_sampler.log_posterior(weight) + bias_log_posterior
        )
        self.log_prior = self.weight_prior_dist.log_prior(weight) + bias_log_prior

        x = weight.unsqueeze(0) * X.unsqueeze(2) + bias

        if self.activation_fn is not None:
            x = self.activation_fn(x)

        return x

    def extra_repr(self) -> str:
        s = "{n_cont_cols}, {embed_dim}, use_bias={use_bias}"
        return s.format(**self.__dict__)


class BayesianDiffSizeCatEmbeddings(nn.Module):
    def __init__(
        self,
        column_idx: Dict[str, int],
        embed_input: List[Tuple[str, int, int]],
        prior_sigma_1: float,
        prior_sigma_2: float,
        prior_pi: float,
        posterior_mu_init: float,
        posterior_rho_init: float,
        activation_fn: Optional[str] = None,
    ):
        super(BayesianDiffSizeCatEmbeddings, self).__init__()

        self.column_idx = column_idx
        self.embed_input = embed_input

        # Categorical: val + 1 because 0 is reserved for padding/unseen cateogories.
        self.embed_layers = nn.ModuleDict(
            {
                "emb_layer_"
                + col: bnn.BayesianEmbedding(
                    val + 1,
                    dim,
                    padding_idx=0,
                    prior_sigma_1=prior_sigma_1,
                    prior_sigma_2=prior_sigma_2,
                    prior_pi=prior_pi,
                    posterior_mu_init=posterior_mu_init,
                    posterior_rho_init=posterior_rho_init,
                )
                for col, val, dim in self.embed_input
            }
        )

        self.emb_out_dim: int = int(np.sum([embed[2] for embed in self.embed_input]))

        self.activation_fn = (
            get_activation_fn(activation_fn) if activation_fn is not None else None
        )

    def forward(self, X: Tensor) -> Tensor:
        embed = [
            self.embed_layers["emb_layer_" + col](X[:, self.column_idx[col]].long())
            for col, _, _ in self.embed_input
        ]
        x = torch.cat(embed, 1)

        if self.activation_fn is not None:
            x = self.activation_fn(x)

        return x
