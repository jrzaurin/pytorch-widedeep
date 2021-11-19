import numpy as np
import einops
import torch.nn.functional as F
from torch import nn

from pytorch_widedeep.wdtypes import *  # noqa: F403
from pytorch_widedeep.models._get_activation_fn import get_activation_fn
from pytorch_widedeep.bayesian_models._weight_sampler import (
    GaussianPosterior,
    ScaleMixtureGaussianPrior,
)
from pytorch_widedeep.bayesian_models._base_bayesian_model import (
    BayesianModule,
)


class BayesianEmbedding(BayesianModule):
    def __init__(
        self,
        n_embed: int,
        embed_dim: int,
        padding_idx: Optional[int] = None,
        max_norm: Optional[float] = None,
        norm_type: Optional[float] = 2.0,
        scale_grad_by_freq: Optional[bool] = False,
        sparse: Optional[bool] = False,
        use_bias: bool = False,
        prior_sigma_1: float = 0.1,
        prior_sigma_2: float = 0.002,
        prior_pi: float = 1.0,
        posterior_mu_init: float = 0.0,
        posterior_rho_init: float = -6.0,
    ):
        super(BayesianEmbedding, self).__init__()

        self.n_embed = n_embed
        self.embed_dim = embed_dim
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self.sparse = sparse

        self.use_bias = use_bias

        self.prior_sigma_1 = prior_sigma_1
        self.prior_sigma_2 = prior_sigma_2
        self.prior_pi = prior_pi
        self.posterior_mu_init = posterior_mu_init
        self.posterior_rho_init = posterior_rho_init

        # Variational weight parameters and sample
        self.weight_mu = nn.Parameter(
            torch.Tensor(n_embed, embed_dim).normal_(posterior_mu_init, 0.1)
        )
        self.weight_rho = nn.Parameter(
            torch.Tensor(n_embed, embed_dim).normal_(posterior_rho_init, 0.1)
        )
        self.weight_sampler = GaussianPosterior(self.weight_mu, self.weight_rho)

        if self.use_bias:
            self.bias_mu: Union[nn.Parameter, float] = nn.Parameter(
                torch.Tensor(n_embed).normal_(posterior_mu_init, 0.1)
            )
            self.bias_rho: Union[nn.Parameter, float] = nn.Parameter(
                torch.Tensor(n_embed).normal_(posterior_rho_init, 0.1)
            )
            self.bias_sampler = GaussianPosterior(self.bias_mu, self.bias_rho)
        else:
            self.bias_mu, self.bias_rho = 0.0, 0.0

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
            return (
                F.embedding(
                    X,
                    self.weight_mu,
                    self.padding_idx,
                    self.max_norm,
                    self.norm_type,
                    self.scale_grad_by_freq,
                    self.sparse,
                )
                + self.bias_mu
            )

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

        return (
            F.embedding(
                X,
                weight,
                self.padding_idx,
                self.max_norm,
                self.norm_type,
                self.scale_grad_by_freq,
                self.sparse,
            )
            + bias
        )

    def extra_repr(self) -> str:  # noqa: C901
        s = "{n_embed}, {embed_dim}"
        if self.padding_idx is not None:
            s += ", padding_idx={padding_idx}"
        if self.max_norm is not None:
            s += ", max_norm={max_norm}"
        if self.norm_type != 2:
            s += ", norm_type={norm_type}"
        if self.scale_grad_by_freq is not False:
            s += ", scale_grad_by_freq={scale_grad_by_freq}"
        if self.sparse is not False:
            s += ", sparse=True"
        if self.use_bias:
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
        activation: str = None,
    ):
        super(BayesianContEmbeddings, self).__init__()

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

        self.act_fn = get_activation_fn(activation) if activation else None

        self.log_prior: Union[Tensor, float] = 0.0
        self.log_variational_posterior: Union[Tensor, float] = 0.0

    def forward(self, X: Tensor) -> Tensor:

        if not self.training:
            x = self.weight_mu.unsqueeze(0) * X.unsqueeze(2)
            if self.bias_mu is not None:
                x + self.bias_mu.unsqueeze(0)
            if self.act_fn is not None:
                x = self.act_fn(x)
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
        if self.act_fn is not None:
            x = self.act_fn(x)

        return x

    def extra_repr(self) -> str:
        s = "{n_cont_cols}, {embed_dim}, embed_dropout={embed_dropout}, use_bias={use_bias}"
        if self.activation is not None:
            s += ", activation={activation}"
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
                + col: BayesianEmbedding(
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
        cont_embed_dim: int,
        cont_embed_activation: str,
        use_cont_bias: bool,
        cont_norm_layer: str,
        prior_sigma_1: float,
        prior_sigma_2: float,
        prior_pi: float,
        posterior_mu_init: float,
        posterior_rho_init: float,
    ):
        super(BayesianDiffSizeCatAndContEmbeddings, self).__init__()

        self.cat_embed_input = cat_embed_input
        self.continuous_cols = continuous_cols

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
            self.cont_embed = BayesianContEmbeddings(
                len(continuous_cols),
                cont_embed_dim,
                prior_sigma_1,
                prior_sigma_2,
                prior_pi,
                posterior_mu_init,
                posterior_rho_init,
                use_cont_bias,
                cont_embed_activation,
            )
            self.cont_out_dim = len(continuous_cols) * cont_embed_dim
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
