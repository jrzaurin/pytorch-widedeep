import numpy as np
import einops
from torch import nn

from pytorch_widedeep.wdtypes import *  # noqa: F403
from pytorch_widedeep.bayesian_models import bayesian_nn as bnn
from pytorch_widedeep.bayesian_models._weight_sampler import (
    GaussianPosterior,
    ScaleMixtureGaussianPrior,
)
from pytorch_widedeep.bayesian_models._base_bayesian_model import (
    BayesianModule,
)


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

    def forward(self, X: Tensor) -> Tensor:
        embed = [
            self.embed_layers["emb_layer_" + col](X[:, self.column_idx[col]].long())
            for col, _, _ in self.embed_input
        ]
        x = torch.cat(embed, 1)
        return x


class BayesianDiffSizeCatAndContEmbeddings(nn.Module):
    def __init__(
        self,
        column_idx: Dict[str, int],
        cat_embed_input: List[Tuple[str, int, int]],
        continuous_cols: Optional[List[str]],
        embed_continuous: bool,
        cont_embed_dim: int,
        use_cont_bias: bool,
        cont_norm_layer: Optional[str],
        prior_sigma_1: float,
        prior_sigma_2: float,
        prior_pi: float,
        posterior_mu_init: float,
        posterior_rho_init: float,
    ):
        super(BayesianDiffSizeCatAndContEmbeddings, self).__init__()

        self.cat_embed_input = cat_embed_input
        self.continuous_cols = continuous_cols
        self.embed_continuous = embed_continuous
        self.cont_embed_dim = cont_embed_dim

        # Categorical
        if self.cat_embed_input is not None:
            self.cat_embed = BayesianDiffSizeCatEmbeddings(
                column_idx,
                cat_embed_input,
                prior_sigma_1,
                prior_sigma_2,
                prior_pi,
                posterior_mu_init,
                posterior_rho_init,
            )
            self.cat_out_dim = int(np.sum([embed[2] for embed in self.cat_embed_input]))
        else:
            self.cat_out_dim = 0

        # Continuous
        if continuous_cols is not None:
            self.cont_idx = [column_idx[col] for col in continuous_cols]
            if cont_norm_layer == "layernorm":
                self.cont_norm: NormLayers = nn.LayerNorm(len(continuous_cols))
            elif cont_norm_layer == "batchnorm":
                self.cont_norm = nn.BatchNorm1d(len(continuous_cols))
            else:
                self.cont_norm = nn.Identity()
            if self.embed_continuous:
                self.cont_embed = BayesianContEmbeddings(
                    len(continuous_cols),
                    cont_embed_dim,
                    prior_sigma_1,
                    prior_sigma_2,
                    prior_pi,
                    posterior_mu_init,
                    posterior_rho_init,
                    use_cont_bias,
                )
                self.cont_out_dim = len(continuous_cols) * cont_embed_dim
            else:
                self.cont_out_dim = len(continuous_cols)
        else:
            self.cont_out_dim = 0

        self.output_dim = self.cat_out_dim + self.cont_out_dim

    def forward(self, X: Tensor) -> Tuple[Tensor, Any]:

        if self.cat_embed_input is not None:
            x_cat = self.cat_embed(X)
        else:
            x_cat = None

        if self.continuous_cols is not None:
            x_cont = self.cont_norm((X[:, self.cont_idx].float()))
            if self.embed_continuous:
                x_cont = self.cont_embed(x_cont)
                x_cont = einops.rearrange(x_cont, "b s d -> b (s d)")
        else:
            x_cont = None

        return x_cat, x_cont
