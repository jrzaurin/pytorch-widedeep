import torch
from torch import nn

from pytorch_widedeep.wdtypes import *  # noqa: F403


class BayesianModule(nn.Module):
    r"""Simply a 'hack' to facilitate the computation of the KL divergence for all
    Bayesian models
    """

    def init(self):
        super().__init__()


class BaseBayesianModel(nn.Module):
    r"""Base model containing the two methods common to all Bayesian models"""

    def init(self):
        super().__init__()

    def _kl_divergence(self):
        kld = 0
        for module in self.modules():
            if isinstance(module, BayesianModule):
                kld += module.log_variational_posterior - module.log_prior
        return kld

    def sample_elbo(
        self,
        input: Tensor,
        target: Tensor,
        loss_fn: nn.Module,
        n_samples: int,
        n_batches: int,
    ) -> Tuple[Tensor, Tensor]:

        outputs_l = []
        kld = 0.0
        for _ in range(n_samples):
            outputs_l.append(self(input))
            kld += self._kl_divergence()
        outputs = torch.stack(outputs_l)

        complexity_cost = kld / n_batches
        likelihood_cost = loss_fn(outputs.mean(0), target)
        return outputs, complexity_cost + likelihood_cost
