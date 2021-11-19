from torch import nn

from pytorch_widedeep.wdtypes import *  # noqa: F403


class BayesianModule(nn.Module):
    def init(self):
        super().__init__()


class BaseBayesianModel(nn.Module):
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
        pred_dim: int,
    ) -> Tensor:
        outputs = torch.zeros(n_samples, target.shape[0], pred_dim)
        kld = 0.0
        for i in range(n_samples):
            outputs[i] = self(input)
            kld += self._kl_divergence()
        complexity_cost = kld / n_batches
        likelihood_cost = loss_fn(outputs.mean(0), target)
        return complexity_cost + likelihood_cost
