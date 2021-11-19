import torch

from pytorch_widedeep.wdtypes import *  # noqa: F403
from pytorch_widedeep.models._get_activation_fn import allowed_activations
from pytorch_widedeep.bayesian_models._base_bayesian_model import (
    BaseBayesianModel,
)
from pytorch_widedeep.bayesian_models.bayesian_embeddings_layers import (
    BayesianDiffSizeCatAndContEmbeddings,
)
from pytorch_widedeep.bayesian_models.tabular.bayesian_mlp._layers import (
    BayesianMLP,
)


class BayesianTabMlp(BaseBayesianModel):
    def __init__(
        self,
        column_idx: Dict[str, int],
        cat_embed_input: Optional[List[Tuple[str, int, int]]] = None,
        cat_embed_dropout: float = 0.1,
        continuous_cols: Optional[List[str]] = None,
        embed_continuous: bool = False,
        cont_embed_dim: int = 32,
        cont_embed_dropout: float = 0.1,
        cont_embed_activation: Optional[str] = None,
        use_cont_bias: bool = True,
        cont_norm_layer: str = "batchnorm",
        mlp_hidden_dims: List[int] = [200, 100],
        mlp_activation: str = "relu",
        use_bias: bool = True,
        prior_sigma_1: float = 0.75,
        prior_sigma_2: float = 0.1,
        prior_pi: float = 0.25,
        posterior_mu_init: float = 0.1,
        posterior_rho_init: float = -3.0,
        pred_dim=1,  # Bayesian models will require their own trainer and need the output layer
    ):
        super(BayesianTabMlp, self).__init__()

        self.column_idx = column_idx
        self.cat_embed_input = cat_embed_input
        self.cat_embed_dropout = cat_embed_dropout

        self.continuous_cols = continuous_cols
        self.embed_continuous = embed_continuous
        self.cont_embed_dim = cont_embed_dim
        self.cont_embed_dropout = cont_embed_dropout
        self.cont_embed_activation = cont_embed_activation
        self.use_cont_bias = use_cont_bias
        self.cont_norm_layer = cont_norm_layer

        self.mlp_hidden_dims = mlp_hidden_dims
        self.mlp_activation = mlp_activation
        self.use_bias = use_bias
        self.prior_sigma_1 = prior_sigma_1
        self.prior_sigma_2 = prior_sigma_2
        self.prior_pi = prior_pi
        self.posterior_mu_init = posterior_mu_init
        self.posterior_rho_init = posterior_rho_init

        self.pred_dim = pred_dim

        if self.mlp_activation not in allowed_activations:
            raise ValueError(
                "Currently, only the following activation functions are supported "
                "for for the MLP's dense layers: {}. Got {} instead".format(
                    ", ".join(allowed_activations), self.mlp_activation
                )
            )

        self.bayesian_cat_and_cont_embed = BayesianDiffSizeCatAndContEmbeddings(
            column_idx,
            cat_embed_input,
            continuous_cols,
            cont_embed_dim,
            cont_embed_activation,
            use_cont_bias,
            cont_norm_layer,
            prior_sigma_1,
            prior_sigma_2,
            prior_pi,
            posterior_mu_init,
            posterior_rho_init,
        )

        mlp_input_dim = self.bayesian_cat_and_cont_embed.output_dim
        mlp_hidden_dims = [mlp_input_dim] + mlp_hidden_dims + [pred_dim]
        self.bayesian_tab_mlp = BayesianMLP(
            mlp_hidden_dims,
            mlp_activation,
            use_bias,
            prior_sigma_1,
            prior_sigma_2,
            prior_pi,
            posterior_mu_init,
            posterior_rho_init,
        )

        # the output_dim attribute will be used as input_dim when "merging" the models
        self.output_dim = mlp_hidden_dims[-1]

    def forward(self, X: Tensor) -> Tensor:
        x_emb, x_cont = self.bayesian_cat_and_cont_embed(X)
        if x_emb is not None:
            x = x_emb
        if x_cont is not None:
            x = torch.cat([x, x_cont], 1) if x_emb is not None else x_cont
        return self.bayesian_tab_mlp(x)
