import numpy as np
import torch
import einops
from torch import nn

from pytorch_widedeep.wdtypes import (
    Dict,
    List,
    Tuple,
    Tensor,
    Literal,
    Optional,
)
from pytorch_widedeep.bayesian_models._base_bayesian_model import (
    BaseBayesianModel,
)
from pytorch_widedeep.bayesian_models.tabular.bayesian_mlp._layers import (
    BayesianMLP,
)
from pytorch_widedeep.bayesian_models.tabular.bayesian_embeddings_layers import (
    NormLayers,
    BayesianContEmbeddings,
    BayesianDiffSizeCatEmbeddings,
)


class BayesianTabMlp(BaseBayesianModel):
    r"""Defines a `BayesianTabMlp` model.

    This class combines embedding representations of the categorical features
    with numerical (aka continuous) features, embedded or not. These are then
    passed through a series of probabilistic dense layers (i.e. a MLP).

    Parameters
    ----------
    column_idx: Dict
        Dict containing the index of the columns that will be passed through
        the `TabMlp` model. Required to slice the tensors. e.g. _{'education':
        0, 'relationship': 1, 'workclass': 2, ...}_
    cat_embed_input: List, Optional, default = None
        List of Tuples with the column name, number of unique values and
        embedding dimension. e.g. _[(education, 11, 32), ...]_
    cat_embed_activation: Optional, str, default = None,
        Activation function for the categorical embeddings, if any. Currently
        _'tanh'_, _'relu'_, _'leaky_relu'_ and _'gelu'_ are supported
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
        _'tanh'_, _'relu'_, _'leaky_relu'_ and _'gelu'_ are supported
    mlp_hidden_dims: List, default = [200, 100]
        List with the number of neurons per dense layer in the mlp.
    mlp_activation: str, default = "relu"
        Activation function for the dense layers of the MLP. Currently
        _'tanh'_, _'relu'_, _'leaky_relu'_ and _'gelu'_ are supported
    prior_sigma_1: float, default = 1.0
        The prior weight distribution is a scaled mixture of two Gaussian
        densities:

        $$
           \begin{aligned}
           P(\mathbf{w}) = \prod_{i=j} \pi N (\mathbf{w}_j | 0, \sigma_{1}^{2}) + (1 - \pi) N (\mathbf{w}_j | 0, \sigma_{2}^{2})
           \end{aligned}
        $$

        `prior_sigma_1` is the prior of the sigma parameter for the first of the two
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

        $$
           \begin{aligned}
           \mathbf{w} &= \mu + log(1 + exp(\rho))
           \end{aligned}
        $$
        where:

        $$
           \begin{aligned}
           \mathcal{N}(x\vert \mu, \sigma) &= \frac{1}{\sqrt{2\pi}\sigma}e^{-\frac{(x-\mu)^2}{2\sigma^2}}\\
           \log{\mathcal{N}(x\vert \mu, \sigma)} &= -\log{\sqrt{2\pi}} -\log{\sigma} -\frac{(x-\mu)^2}{2\sigma^2}\\
           \end{aligned}
        $$

        $\mu$ is initialised using a normal distributtion with mean
        `posterior_mu_init` and std equal to 0.1.
    posterior_rho_init: float = -7.0
        As in the case of $\mu$, $\rho$ is initialised using a
        normal distributtion with mean `posterior_rho_init` and std equal to
        0.1.

    Attributes
    ----------
    bayesian_cat_and_cont_embed: nn.Module
        This is the module that processes the categorical and continuous columns
    bayesian_tab_mlp: nn.Sequential
        mlp model that will receive the concatenation of the embeddings and
        the continuous columns

    Examples
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
        *,
        cat_embed_input: Optional[List[Tuple[str, int, int]]] = None,
        cat_embed_activation: Optional[str] = None,
        continuous_cols: Optional[List[str]] = None,
        embed_continuous: Optional[bool] = None,
        cont_embed_dim: Optional[int] = None,
        cont_embed_dropout: Optional[float] = None,
        cont_embed_activation: Optional[str] = None,
        use_cont_bias: Optional[bool] = None,
        cont_norm_layer: Optional[Literal["batchnorm", "layernorm"]] = None,
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

        # Categorical
        if self.cat_embed_input is not None:
            self.cat_embed = BayesianDiffSizeCatEmbeddings(
                column_idx=self.column_idx,
                embed_input=self.cat_embed_input,
                prior_sigma_1=self.prior_sigma_1,
                prior_sigma_2=self.prior_sigma_2,
                prior_pi=self.prior_pi,
                posterior_mu_init=self.posterior_mu_init,
                posterior_rho_init=self.posterior_rho_init,
                activation_fn=self.cat_embed_activation,
            )
            self.cat_out_dim = int(np.sum([embed[2] for embed in self.cat_embed_input]))
        else:
            self.cat_out_dim = 0

        # Continuous
        if self.continuous_cols is not None:
            self.cont_idx = [self.column_idx[col] for col in self.continuous_cols]
            if cont_norm_layer == "layernorm":
                self.cont_norm: NormLayers = nn.LayerNorm(len(self.continuous_cols))
            elif cont_norm_layer == "batchnorm":
                self.cont_norm = nn.BatchNorm1d(len(self.continuous_cols))
            else:
                self.cont_norm = nn.Identity()
            if self.embed_continuous:
                assert self.cont_embed_dim is not None, (
                    "If 'embed_continuous' is True, 'cont_embed_dim' must be "
                    "provided"
                )
                self.cont_embed = BayesianContEmbeddings(
                    n_cont_cols=len(self.continuous_cols),
                    embed_dim=self.cont_embed_dim,
                    prior_sigma_1=self.prior_sigma_1,
                    prior_sigma_2=self.prior_sigma_2,
                    prior_pi=self.prior_pi,
                    posterior_mu_init=self.posterior_mu_init,
                    posterior_rho_init=self.posterior_rho_init,
                    use_bias=(
                        False if self.use_cont_bias is None else self.use_cont_bias
                    ),
                    activation_fn=self.cont_embed_activation,
                )
                self.cont_out_dim = len(self.continuous_cols) * self.cont_embed_dim
            else:
                self.cont_out_dim = len(self.continuous_cols)
        else:
            self.cont_out_dim = 0

        self.output_dim = self.cat_out_dim + self.cont_out_dim

        mlp_hidden_dims = [self.output_dim] + mlp_hidden_dims + [pred_dim]
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
        x = self._get_embeddings(X)
        x = self.bayesian_tab_mlp(x)
        return x

    def _get_embeddings(self, X: Tensor) -> Tensor:
        tensors_to_concat: List[Tensor] = []
        if self.cat_embed_input is not None:
            x_cat = self.cat_embed(X)
            tensors_to_concat.append(x_cat)

        if self.continuous_cols is not None:
            x_cont = self.cont_norm((X[:, self.cont_idx].float()))
            if self.embed_continuous:
                x_cont = self.cont_embed(x_cont)
                x_cont = einops.rearrange(x_cont, "b s d -> b (s d)")
            tensors_to_concat.append(x_cont)

        x = torch.cat(tensors_to_concat, 1)

        return x
