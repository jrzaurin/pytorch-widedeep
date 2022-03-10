from torch import nn

from pytorch_widedeep.wdtypes import *  # noqa: F403
from pytorch_widedeep.bayesian_models import bayesian_nn as bnn
from pytorch_widedeep.bayesian_models._base_bayesian_model import (
    BaseBayesianModel,
)


class BayesianWide(BaseBayesianModel):
    r"""Defines a ``Wide`` model. This is a linear model where the
    non-linearlities are captured via crossed-columns

    Parameters
    ----------
    input_dim: int
        size of the Embedding layer. ``input_dim`` is the summation of all the
        individual values for all the features that go through the wide
        component. For example, if the wide component receives 2 features with
        5 individual values each, `input_dim = 10`
    pred_dim: int
        size of the ouput tensor containing the predictions
    prior_sigma_1: float, default = 1.0
        The prior weight distribution is a scaled mixture of two Gaussian
        densities:

        .. math::
           \begin{aligned}
           P(\mathbf{w}) = \prod_{i=j} \pi N (\mathbf{w}_j | 0, \sigma_{1}^{2}) + (1 - \pi) N (\mathbf{w}_j | 0, \sigma_{2}^{2})
           \end{aligned}

        This is the prior of the sigma parameter for the first of the two
        Gaussians that will be mixed to produce the prior weight
        distribution.
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

    Attributes
    -----------
    bayesian_wide_linear: ``nn.Module``
        the linear layer that comprises the wide branch of the model

    Examples
    --------
    >>> import torch
    >>> from pytorch_widedeep.bayesian_models import BayesianWide
    >>> X = torch.empty(4, 4).random_(6)
    >>> wide = BayesianWide(input_dim=X.unique().size(0), pred_dim=1)
    >>> out = wide(X)
    """

    def __init__(
        self,
        input_dim: int,
        pred_dim: int = 1,
        prior_sigma_1: float = 1.0,
        prior_sigma_2: float = 0.002,
        prior_pi: float = 0.8,
        posterior_mu_init: float = 0.0,
        posterior_rho_init: float = -7.0,
    ):
        super(BayesianWide, self).__init__()
        #  Embeddings: val + 1 because 0 is reserved for padding/unseen cateogories.
        self.bayesian_wide_linear = bnn.BayesianEmbedding(
            n_embed=input_dim + 1,
            embed_dim=pred_dim,
            padding_idx=0,
            prior_sigma_1=prior_sigma_1,
            prior_sigma_2=prior_sigma_2,
            prior_pi=prior_pi,
            posterior_mu_init=posterior_mu_init,
            posterior_rho_init=posterior_rho_init,
        )
        self.bias = nn.Parameter(torch.zeros(pred_dim))

    def forward(self, X: Tensor) -> Tensor:
        out = self.bayesian_wide_linear(X.long()).sum(dim=1) + self.bias
        return out
