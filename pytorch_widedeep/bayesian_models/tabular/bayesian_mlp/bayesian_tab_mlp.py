import torch

from pytorch_widedeep.wdtypes import *  # noqa: F403
from pytorch_widedeep.models._get_activation_fn import allowed_activations
from pytorch_widedeep.bayesian_models._base_bayesian_model import (
    BaseBayesianModel,
)
from pytorch_widedeep.bayesian_models.tabular.bayesian_mlp._layers import (
    BayesianMLP,
)
from pytorch_widedeep.bayesian_models.tabular.bayesian_embeddings_layers import (
    BayesianDiffSizeCatAndContEmbeddings,
)


class BayesianTabMlp(BaseBayesianModel):
    r"""Defines a ``TabMlp`` model that can be used as the ``deeptabular``
    component of a Wide & Deep model.

    This class combines embedding representations of the categorical features
    with numerical (aka continuous) features. These are then passed through a
    series of dense layers (i.e. a MLP).

    Parameters
    ----------
    column_idx: Dict
        Dict containing the index of the columns that will be passed through
        the ``TabMlp`` model. Required to slice the tensors. e.g. {'education':
        0, 'relationship': 1, 'workclass': 2, ...}
    cat_embed_input: List, Optional, default = None
        List of Tuples with the column name, number of unique values and
        embedding dimension. e.g. [(education, 11, 32), ...]
    cat_embed_dropout: float, default = 0.1
        embeddings dropout
    continuous_cols: List, Optional, default = None
        List with the name of the numeric (aka continuous) columns
    embed_continuous: bool, default = False,
        Boolean indicating if the continuous columns will be embedded
        (i.e. passed each through a linear layer with or without activation)
    cont_embed_dim: int, default = 32,
        Size of the continuous embeddings
    cont_embed_dropout: float, default = 0.1,
        Dropout for the continuous embeddings
    cont_embed_activation: Optional, str, default = None,
        Activation function for the continuous embeddings
    use_cont_bias: bool, default = True,
        Boolean indicating in bias will be used for the continuous embeddings
    cont_norm_layer: str, default =  "batchnorm"
        Type of normalization layer applied to the continuous features. Options
        are: 'layernorm', 'batchnorm' or None.
    mlp_hidden_dims: List, default = [200, 100]
        List with the number of neurons per dense layer in the mlp.
    mlp_activation: str, default = "relu"
        Activation function for the dense layers of the MLP. Currently
        ``tanh``, ``relu``, ``leaky_relu`` and ``gelu`` are supported
    prior_sigma_1: float, default = 1.0
        Prior of the sigma parameter for the first of the two weight Gaussian
        distributions that will be mixed to produce the prior weight
        distribution for each Bayesian linear and embedding layer
    prior_sigma_2: float, default = 0.002
        Prior of the sigma parameter for the second of the two weight Gaussian
        distributions that will be mixed to produce the prior weight
        distribution for each Bayesian linear and embedding layer
    prior_pi: float, default = 0.8
        Scaling factor that will be used to mix the Gaussians to produce the
        prior weight distribution ffor each Bayesian linear and embedding
        layer
    posterior_mu_init: float = 0.0,
        The posterior sample of the weights is defined as:

            :math:`\mathbf{w} = \mu + log(1 + exp(\rho))`

        where :math:`\mu` and :math:`\rho` are both sampled from Gaussian
        distributions. ``posterior_mu_init`` is the initial mean value for
        the Gaussian distribution from which :math:`\mu` is sampled for each
        Bayesian linear and embedding layers.
    posterior_rho_init: float = -7.0,
        The initial mean value for the Gaussian distribution from
        which :math:`\rho` is sampled for each Bayesian linear and embedding
        layers.

    Attributes
    ----------
    bayesian_cat_and_cont_embed: ``nn.Module``
        This is the module that processes the categorical and continuous columns
    bayesian_tab_mlp: ``nn.Sequential``
        mlp model that will receive the concatenation of the embeddings and
        the continuous columns

    Example
    --------
    >>> import torch
    >>> from pytorch_widedeep.bayesian_models import BayesianTabMlp
    >>> X_tab = torch.cat((torch.empty(5, 4).random_(4), torch.rand(5, 1)), axis=1)
    >>> colnames = ['a', 'b', 'c', 'd', 'e']
    >>> cat_embed_input = [(u,i,j) for u,i,j in zip(colnames[:4], [4]*4, [8]*4)]
    >>> column_idx = {k:v for v,k in enumerate(colnames)}
    >>> model = BayesianTabMlp(mlp_hidden_dims=[8,4], column_idx=column_idx, cat_embed_input=cat_embed_input,
    ... continuous_cols = ['e'])
    >>> out = model(X_tab)
    """

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
        mlp_activation: str = "leaky_relu",
        prior_sigma_1: float = 0.75,
        prior_sigma_2: float = 0.1,
        prior_pi: float = 0.25,
        posterior_mu_init: float = 0.0,
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
            embed_continuous,
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
            True,  # use_bias
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
