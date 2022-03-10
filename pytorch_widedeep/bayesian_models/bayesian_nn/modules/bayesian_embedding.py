"""
The code here is greatly insipired by the code at the Blitz package:

https://github.com/piEsposito/blitz-bayesian-deep-learning
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


class BayesianEmbedding(BayesianModule):
    r"""A simple lookup table that looks up embeddings in a fixed dictionary and
    size.

    Parameters
    ----------
    n_embed: int
        number of embeddings. Typically referred as size of the vocabulary
    embed_dim: int
        Dimension of the embeddings
    padding_idx: int, optional, default = None
        If specified, the entries at ``padding_idx`` do not contribute to the
        gradient; therefore, the embedding vector at ``padding_idx`` is not
        updated during training, i.e. it remains as a fixed “pad”. For a
        newly constructed Embedding, the embedding vector at ``padding_idx``
        will default to all zeros, but can be updated to another value to be
        used as the padding vector
    max_norm: float, optional, default = None
        If given, each embedding vector with norm larger than ``max_norm`` is
        renormalized to have norm max_norm
    norm_type: float, optional, default = 2.
        The p of the p-norm to compute for the ``max_norm`` option.
    scale_grad_by_freq: bool, optional, default = False
        If given, this will scale gradients by the inverse of frequency of the
        words in the mini-batch.
    sparse: bool, optional, default = False
        If True, gradient w.r.t. weight matrix will be a sparse tensor. See
        Notes for more details regarding sparse gradients.
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
    >>> embedding = bnn.BayesianEmbedding(10, 3)
    >>> input = torch.LongTensor([[1,2,4,5],[4,3,2,9]])
    >>> out = embedding(input)
    """

    def __init__(
        self,
        n_embed: int,
        embed_dim: int,
        padding_idx: Optional[int] = None,
        max_norm: Optional[float] = None,
        norm_type: Optional[float] = 2.0,
        scale_grad_by_freq: Optional[bool] = False,
        sparse: Optional[bool] = False,
        prior_sigma_1: float = 1.0,
        prior_sigma_2: float = 0.002,
        prior_pi: float = 0.8,
        posterior_mu_init: float = 0.0,
        posterior_rho_init: float = -7.0,
    ):
        super(BayesianEmbedding, self).__init__()

        self.n_embed = n_embed
        self.embed_dim = embed_dim
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self.sparse = sparse

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

        # Prior
        self.weight_prior_dist = ScaleMixtureGaussianPrior(
            self.prior_pi,
            self.prior_sigma_1,
            self.prior_sigma_2,
        )

        self.log_prior: Union[Tensor, float] = 0.0
        self.log_variational_posterior: Union[Tensor, float] = 0.0

    def forward(self, X: Tensor) -> Tensor:

        if not self.training:
            return F.embedding(
                X,
                self.weight_mu,
                self.padding_idx,
                self.max_norm,
                self.norm_type,
                self.scale_grad_by_freq,
                self.sparse,
            )

        weight = self.weight_sampler.sample()

        self.log_variational_posterior = self.weight_sampler.log_posterior(weight)
        self.log_prior = self.weight_prior_dist.log_prior(weight)

        return F.embedding(
            X,
            weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
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
