from pytorch_widedeep.wdtypes import *  # noqa: F403
from pytorch_widedeep.bayesian_models._base_bayesian_model import (
    BaseBayesianModel,
)
from pytorch_widedeep.bayesian_models.bayesian_embeddings_layers import (
    BayesianEmbedding,
)


class BayesianWide(BaseBayesianModel):
    def __init__(
        self,
        input_dim: int,
        pred_dim: int = 1,
        prior_sigma_1: float = 0.75,
        prior_sigma_2: float = 1,
        prior_pi: float = 0.25,
        posterior_mu_init: float = 0.1,
        posterior_rho_init: float = -3.0,
    ):
        super(BayesianWide, self).__init__()
        self.bayesian_wide_linear = BayesianEmbedding(
            n_embed=input_dim,
            embed_dim=pred_dim,
            padding_idx=0,
            use_bias=True,
            prior_sigma_1=prior_sigma_1,
            prior_sigma_2=prior_sigma_2,
            prior_pi=prior_pi,
            posterior_mu_init=posterior_mu_init,
            posterior_rho_init=posterior_rho_init,
        )

    def forward(self, X: Tensor) -> Tensor:
        out = self.bayesian_wide_linear(X.long()).sum(dim=1)
        return out
