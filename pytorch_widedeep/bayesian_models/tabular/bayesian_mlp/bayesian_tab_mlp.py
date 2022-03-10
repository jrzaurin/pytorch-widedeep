import torch

from pytorch_widedeep.wdtypes import *  # noqa: F403
from pytorch_widedeep.models._get_activation_fn import get_activation_fn
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
    r"""Defines a ``BayesianTabMlp`` model.

    This class combines embedding representations of the categorical features
    with numerical (aka continuous) features, embedded or not. These are then
    passed through a series of dense layers (i.e. a MLP).

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
        Categorical embeddings dropout
    cat_embed_activation: Optional, str, default = None,
        Activation function for the categorical embeddings, if any. Currently
        `'tanh'`, `'relu'`, `'leaky_relu'` and `'gelu'` are supported
    continuous_cols: List, Optional, default = None
        List with the name of the numeric (aka continuous) columns
    cont_norm_layer: str, default =  "batchnorm"
        Type of normalization layer applied to the continuous features. Options
        are: 'layernorm', 'batchnorm' or None.
    embed_continuous: bool, default = False,
        Boolean indicating if the continuous columns will be embedded
        (i.e. passed each through a linear layer with or without activation)
    cont_embed_dim: int, default = 32,
        Size of the continuous embeddings
    cont_embed_dropout: float, default = 0.1,
        Dropout for the continuous embeddings
    use_cont_bias: bool, default = True,
        Boolean indicating if bias will be used for the continuous embeddings
    cont_embed_activation: Optional, str, default = None,
        Activation function for the continuous embeddings if any. Currently
        `'tanh'`, `'relu'`, `'leaky_relu'` and `'gelu'` are supported
    mlp_hidden_dims: List, default = [200, 100]
        List with the number of neurons per dense layer in the mlp.
    mlp_activation: str, default = "relu"
        Activation function for the dense layers of the MLP. Currently
        `'tanh'`, `'relu'`, `'leaky_relu'` and `'gelu'` are supported
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
        distribution for each Bayesian linear and embedding layer
    prior_pi: float, default = 0.8
        Scaling factor that will be used to mix the Gaussians to produce the
        prior weight distribution ffor each Bayesian linear and embedding
        layer
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
        cat_embed_activation: Optional[str] = None,
        continuous_cols: Optional[List[str]] = None,
        embed_continuous: bool = False,
        cont_embed_dim: int = 32,
        cont_embed_dropout: float = 0.1,
        cont_embed_activation: Optional[str] = None,
        use_cont_bias: bool = True,
        cont_norm_layer: str = "batchnorm",
        mlp_hidden_dims: List[int] = [200, 100],
        mlp_activation: str = "leaky_relu",
        prior_sigma_1: float = 1,
        prior_sigma_2: float = 0.002,
        prior_pi: float = 0.8,
        posterior_mu_init: float = 0.0,
        posterior_rho_init: float = -7.0,
        pred_dim=1,  # Bayesian models will require their own trainer and need the output layer
    ):
        super(BayesianTabMlp, self).__init__()

        self.column_idx = column_idx
        self.cat_embed_input = cat_embed_input
        self.cat_embed_dropout = cat_embed_dropout
        self.cat_embed_activation = cat_embed_activation

        self.continuous_cols = continuous_cols
        self.cont_norm_layer = cont_norm_layer
        self.embed_continuous = embed_continuous
        self.cont_embed_dim = cont_embed_dim
        self.cont_embed_dropout = cont_embed_dropout
        self.use_cont_bias = use_cont_bias
        self.cont_embed_activation = cont_embed_activation

        self.mlp_hidden_dims = mlp_hidden_dims
        self.mlp_activation = mlp_activation
        self.prior_sigma_1 = prior_sigma_1
        self.prior_sigma_2 = prior_sigma_2
        self.prior_pi = prior_pi
        self.posterior_mu_init = posterior_mu_init
        self.posterior_rho_init = posterior_rho_init

        self.pred_dim = pred_dim

        allowed_activations = ["relu", "leaky_relu", "tanh", "gelu"]
        if self.mlp_activation not in allowed_activations:
            raise ValueError(
                "Currently, only the following activation functions are supported "
                "for the Bayesian MLP's dense layers: {}. Got '{}' instead".format(
                    ", ".join(allowed_activations),
                    self.mlp_activation,
                )
            )

        self.cat_and_cont_embed = BayesianDiffSizeCatAndContEmbeddings(
            column_idx,
            cat_embed_input,
            continuous_cols,
            embed_continuous,
            cont_embed_dim,
            use_cont_bias,
            cont_norm_layer,
            prior_sigma_1,
            prior_sigma_2,
            prior_pi,
            posterior_mu_init,
            posterior_rho_init,
        )
        self.cat_embed_act_fn = (
            get_activation_fn(cat_embed_activation)
            if cat_embed_activation is not None
            else None
        )
        self.cont_embed_act_fn = (
            get_activation_fn(cont_embed_activation)
            if cont_embed_activation is not None
            else None
        )

        mlp_input_dim = self.cat_and_cont_embed.output_dim
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

    def forward(self, X: Tensor) -> Tensor:
        x_cat, x_cont = self.cat_and_cont_embed(X)
        if x_cat is not None:
            x = (
                self.cat_embed_act_fn(x_cat)
                if self.cat_embed_act_fn is not None
                else x_cat
            )
        if x_cont is not None:
            if self.cont_embed_act_fn is not None:
                x_cont = self.cont_embed_act_fn(x_cont)
            x = torch.cat([x, x_cont], 1) if x_cat is not None else x_cont
        return self.bayesian_tab_mlp(x)
