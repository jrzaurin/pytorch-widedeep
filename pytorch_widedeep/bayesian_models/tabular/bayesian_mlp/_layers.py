from torch import nn

from pytorch_widedeep.wdtypes import *  # noqa: F403
from pytorch_widedeep.models._get_activation_fn import get_activation_fn
from pytorch_widedeep.bayesian_models.bayesian_linear import BayesianLinear


class BayesianMLP(nn.Module):
    def __init__(
        self,
        d_hidden: List[int],
        activation: str,
        use_bias: bool = True,
        prior_sigma_1: float = 0.75,
        prior_sigma_2: float = 0.1,
        prior_pi: float = 0.25,
        posterior_mu_init: float = 0.1,
        posterior_rho_init: float = -3.0,
    ):
        super(BayesianMLP, self).__init__()

        self.d_hidden = d_hidden
        self.activation = activation

        act_fn = get_activation_fn(activation)
        self.bayesian_mlp = nn.Sequential()
        for i in range(1, len(d_hidden)):
            bayesian_dense_layer = nn.Sequential(
                *[
                    BayesianLinear(
                        d_hidden[i - 1],
                        d_hidden[i],
                        use_bias,
                        prior_sigma_1,
                        prior_sigma_2,
                        prior_pi,
                        posterior_mu_init,
                        posterior_rho_init,
                    ),
                    act_fn,
                ]
            )
            self.bayesian_mlp.add_module(
                "bayesian_dense_layer_{}".format(i - 1), bayesian_dense_layer
            )

    def forward(self, X: Tensor) -> Tensor:
        return self.bayesian_mlp(X)