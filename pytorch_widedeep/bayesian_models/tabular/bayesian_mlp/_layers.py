from torch import nn

from pytorch_widedeep.wdtypes import *  # noqa: F403
from pytorch_widedeep.bayesian_models import bayesian_nn as bnn
from pytorch_widedeep.models._get_activation_fn import get_activation_fn


class BayesianMLP(nn.Module):
    def __init__(
        self,
        d_hidden: List[int],
        activation: str,
        use_bias: bool = True,
        prior_sigma_1: float = 1.0,
        prior_sigma_2: float = 0.002,
        prior_pi: float = 0.8,
        posterior_mu_init: float = 0.0,
        posterior_rho_init: float = -7.0,
    ):
        super(BayesianMLP, self).__init__()

        self.d_hidden = d_hidden
        self.activation = activation

        act_fn = get_activation_fn(activation)
        self.bayesian_mlp = nn.Sequential()
        for i in range(1, len(d_hidden)):
            bayesian_dense_layer = nn.Sequential(
                *[
                    bnn.BayesianLinear(
                        d_hidden[i - 1],
                        d_hidden[i],
                        use_bias,
                        prior_sigma_1,
                        prior_sigma_2,
                        prior_pi,
                        posterior_mu_init,
                        posterior_rho_init,
                    ),
                    # The activation of the output neuron(s) will happen
                    # inside the BayesianTrainer
                    act_fn if i != len(d_hidden) - 1 else nn.Identity(),
                ]
            )
            self.bayesian_mlp.add_module(
                "bayesian_dense_layer_{}".format(i - 1), bayesian_dense_layer
            )

    def forward(self, X: Tensor) -> Tensor:
        return self.bayesian_mlp(X)
