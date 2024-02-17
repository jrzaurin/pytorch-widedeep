import torch
from torch import nn

from pytorch_widedeep.wdtypes import (
    Dict,
    List,
    Tuple,
    Tensor,
    Literal,
    Optional,
)
from pytorch_widedeep.models.tabular.tabnet._layers import (
    TabNetEncoder,
    FeatTransformer,
    initialize_non_glu,
)
from pytorch_widedeep.models.tabular._base_tabular_model import (
    BaseTabularModelWithoutAttention,
)


class TabNet(BaseTabularModelWithoutAttention):
    r"""Defines a [TabNet model](https://arxiv.org/abs/1908.07442) that
    can be used as the `deeptabular` component of a Wide & Deep model or
    independently by itself.

    The implementation in this library is fully based on that
    [here](https://github.com/dreamquark-ai/tabnet) by the dreamquark-ai team,
    simply adapted so that it can work within the `WideDeep` frame.
    Therefore, **ALL CREDIT TO THE DREAMQUARK-AI TEAM**.

    Parameters
    ----------
    column_idx: Dict
        Dict containing the index of the columns that will be passed through
        the `TabMlp` model. Required to slice the tensors. e.g. _{'education':
        0, 'relationship': 1, 'workclass': 2, ...}_.
    cat_embed_input: List, Optional, default = None
        List of Tuples with the column name, number of unique values and
        embedding dimension. e.g. _[(education, 11, 32), ...]_
    cat_embed_dropout: float, Optional, default = None
        Categorical embeddings dropout. If `None`, it will default
        to 0.
    use_cat_bias: bool, Optional, default = None,
        Boolean indicating if bias will be used for the categorical embeddings.
        If `None`, it will default to 'False'.
    cat_embed_activation: Optional, str, default = None,
        Activation function for the categorical embeddings, if any. Currently
        _'tanh'_, _'relu'_, _'leaky_relu'_ and _'gelu'_ are supported
    continuous_cols: List, Optional, default = None
        List with the name of the numeric (aka continuous) columns
    cont_norm_layer: str, Optional, default =  None
        Type of normalization layer applied to the continuous features.
        Options are: _'layernorm'_ and _'batchnorm'_. if `None`, no
        normalization layer will be used.
    embed_continuous: bool, Optional, default = None,
        Boolean indicating if the continuous columns will be embedded using
        one of the available methods: _'standard'_, _'periodic'_
        or _'piecewise'_. If `None`, it will default to 'False'.<br/>
        :information_source: **NOTE**: This parameter is deprecated and it
         will be removed in future releases. Please, use the
         `embed_continuous_method` parameter instead.
    embed_continuous_method: Optional, str, default = None,
        Method to use to embed the continuous features. Options are:
        _'standard'_, _'periodic'_ or _'piecewise'_. The _'standard'_
        embedding method is based on the FT-Transformer implementation
        presented in the paper: [Revisiting Deep Learning Models for
        Tabular Data](https://arxiv.org/abs/2106.11959v5). The _'periodic'_
        and_'piecewise'_ methods were presented in the paper: [On Embeddings for
        Numerical Features in Tabular Deep Learning](https://arxiv.org/abs/2203.05556).
        Please, read the papers for details.
    cont_embed_dim: int, Optional, default = None,
        Size of the continuous embeddings. If the continuous columns are
        embedded, `cont_embed_dim` must be passed.
    cont_embed_dropout: float, Optional, default = None,
        Dropout for the continuous embeddings. If `None`, it will default to 0.0
    cont_embed_activation: Optional, str, default = None,
        Activation function for the continuous embeddings if any. Currently
        _'tanh'_, _'relu'_, _'leaky_relu'_ and _'gelu'_ are supported.
        If `None`, no activation function will be applied.
    quantization_setup: Dict[str, List[float]], Optional, default = None,
        This parameter is used when the _'piecewise'_ method is used to embed
        the continuous cols. It is a dict where keys are the name of the continuous
        columns and values are lists with the boundaries for the quantization
        of the continuous_cols. See the examples for details. If
        If the _'piecewise'_ method is used, this parameter is required.
    n_frequencies: int, Optional, default = None,
        This is the so called _'k'_ in their paper [On Embeddings for
        Numerical Features in Tabular Deep Learning](https://arxiv.org/abs/2203.05556),
        and is the number of 'frequencies' that will be used to represent each
        continuous column. See their Eq 2 in the paper for details. If
        the _'periodic'_ method is used, this parameter is required.
    sigma: float, Optional, default = None,
        This is the sigma parameter in the paper mentioned when describing the
        previous parameters and it is used to initialise the 'frequency
        weights'. See their Eq 2 in the paper for details. If
        the _'periodic'_ method is used, this parameter is required.
    share_last_layer: bool, Optional, default = None,
        This parameter is not present in the before mentioned paper but it is implemented in
        the [official repo](https://github.com/yandex-research/rtdl-num-embeddings/tree/main).
        If `True` the linear layer that turns the frequencies into embeddings
        will be shared across the continuous columns. If `False` a different
        linear layer will be used for each continuous column.
        If the _'periodic'_ method is used, this parameter is required.
    full_embed_dropout: bool, Optional, default = None,
        If `True`, the full embedding corresponding to a column will be masked
        out/dropout. If `None`, it will default to `False`.
    n_steps: int, default = 3
        number of decision steps. For a better understanding of the function
        of `n_steps` and the upcoming parameters, please see the
        [paper](https://arxiv.org/abs/1908.07442).
    step_dim: int, default = 8
        Step's output dimension. This is the output dimension that
        `WideDeep` will collect and connect to the output neuron(s).
    attn_dim: int, default = 8
        Attention dimension
    dropout: float, default = 0.0
        GLU block's internal dropout
    n_glu_step_dependent: int, default = 2
        number of GLU Blocks (`[FC -> BN -> GLU]`) that are step dependent
    n_glu_shared: int, default = 2
        number of GLU Blocks (`[FC -> BN -> GLU]`) that will be shared
        across decision steps
    ghost_bn: bool, default=True
        Boolean indicating if [Ghost Batch Normalization](https://arxiv.org/abs/1705.08741)
        will be used.
    virtual_batch_size: int, default = 128
        Batch size when using Ghost Batch Normalization
    momentum: float, default = 0.02
        Ghost Batch Normalization's momentum. The dreamquark-ai advises for
        very low values. However high values are used in the original
        publication. During our tests higher values lead to better results
    gamma: float, default = 1.3
        Relaxation parameter in the paper. When gamma = 1, a feature is
        enforced to be used only at one decision step. As gamma increases,
        more flexibility is provided to use a feature at multiple decision
        steps
    epsilon: float, default = 1e-15
        Float to avoid log(0). Always keep low
    mask_type: str, default = "sparsemax"
        Mask function to use. Either _'sparsemax'_ or _'entmax'_

    Attributes
    ----------
    encoder: nn.Module
        the TabNet encoder. For details see the [original publication](https://arxiv.org/abs/1908.07442).

    Examples
    --------
    >>> X_tab = torch.cat((torch.empty(5, 4).random_(4), torch.rand(5, 1)), axis=1)
    >>> colnames = ["a", "b", "c", "d", "e"]
    >>> cat_embed_input = [(u, i, j) for u, i, j in zip(colnames[:4], [4] * 4, [8] * 4)]
    >>> column_idx = {k: v for v, k in enumerate(colnames)}
    >>> model = TabNet(column_idx=column_idx, cat_embed_input=cat_embed_input, continuous_cols=["e"])
    >>> out = model(X_tab)
    """

    def __init__(
        self,
        column_idx: Dict[str, int],
        *,
        cat_embed_input: Optional[List[Tuple[str, int, int]]] = None,
        cat_embed_dropout: Optional[float] = None,
        use_cat_bias: Optional[bool] = None,
        cat_embed_activation: Optional[str] = None,
        continuous_cols: Optional[List[str]] = None,
        cont_norm_layer: Optional[Literal["batchnorm", "layernorm"]] = None,
        embed_continuous: Optional[bool] = None,
        embed_continuous_method: Optional[
            Literal["standard", "piecewise", "periodic"]
        ] = None,
        cont_embed_dim: Optional[int] = None,
        cont_embed_dropout: Optional[float] = None,
        cont_embed_activation: Optional[str] = None,
        quantization_setup: Optional[Dict[str, List[float]]] = None,
        n_frequencies: Optional[int] = None,
        sigma: Optional[float] = None,
        share_last_layer: Optional[bool] = None,
        full_embed_dropout: Optional[bool] = None,
        n_steps: int = 3,
        step_dim: int = 8,
        attn_dim: int = 8,
        dropout: float = 0.0,
        n_glu_step_dependent: int = 2,
        n_glu_shared: int = 2,
        ghost_bn: bool = True,
        virtual_batch_size: int = 128,
        momentum: float = 0.02,
        gamma: float = 1.3,
        epsilon: float = 1e-15,
        mask_type: str = "sparsemax",
    ):
        super(TabNet, self).__init__(
            column_idx=column_idx,
            cat_embed_input=cat_embed_input,
            cat_embed_dropout=cat_embed_dropout,
            use_cat_bias=use_cat_bias,
            cat_embed_activation=cat_embed_activation,
            continuous_cols=continuous_cols,
            cont_norm_layer=cont_norm_layer,
            embed_continuous=embed_continuous,
            embed_continuous_method=embed_continuous_method,
            cont_embed_dim=cont_embed_dim,
            cont_embed_dropout=cont_embed_dropout,
            cont_embed_activation=cont_embed_activation,
            quantization_setup=quantization_setup,
            n_frequencies=n_frequencies,
            sigma=sigma,
            share_last_layer=share_last_layer,
            full_embed_dropout=full_embed_dropout,
        )

        self.n_steps = n_steps
        self.step_dim = step_dim
        self.attn_dim = attn_dim
        self.dropout = dropout
        self.n_glu_step_dependent = n_glu_step_dependent
        self.n_glu_shared = n_glu_shared
        self.ghost_bn = ghost_bn
        self.virtual_batch_size = virtual_batch_size
        self.momentum = momentum
        self.gamma = gamma
        self.epsilon = epsilon
        self.mask_type = mask_type

        # Embeddings are instantiated at the base model
        self.embed_out_dim = self.cat_out_dim + self.cont_out_dim

        # TabNet
        self.encoder = TabNetEncoder(
            self.embed_out_dim,
            n_steps,
            step_dim,
            attn_dim,
            dropout,
            n_glu_step_dependent,
            n_glu_shared,
            ghost_bn,
            virtual_batch_size,
            momentum,
            gamma,
            epsilon,
            mask_type,
        )

    def forward(
        self, X: Tensor, prior: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        x = self._get_embeddings(X)
        steps_output, M_loss = self.encoder(x, prior)
        res = torch.sum(torch.stack(steps_output, dim=0), dim=0)
        return (res, M_loss)

    def forward_masks(self, X: Tensor) -> Tuple[Tensor, Dict[int, Tensor]]:
        x = self._get_embeddings(X)
        return self.encoder.forward_masks(x)

    @property
    def output_dim(self) -> int:
        r"""The output dimension of the model. This is a required property
        neccesary to build the `WideDeep` class
        """
        return self.step_dim


class TabNetPredLayer(nn.Module):
    def __init__(self, inp, out):
        r"""This class is a 'hack' required because TabNet is a very particular
        model within `WideDeep`.

        TabNet's forward method within `WideDeep` outputs two tensors, one
        with the last layer's activations and the sparse regularization
        factor. Since the output needs to be collected by `WideDeep` to then
        Sequentially build the output layer (connection to the output
        neuron(s)) I need to code a custom TabNetPredLayer that accepts two
        inputs. This will be used by the `WideDeep` class.
        """
        super(TabNetPredLayer, self).__init__()
        self.pred_layer = nn.Linear(inp, out, bias=False)
        initialize_non_glu(self.pred_layer, inp, out)

    def forward(self, tabnet_tuple: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        res, M_loss = tabnet_tuple[0], tabnet_tuple[1]
        return self.pred_layer(res), M_loss


class TabNetDecoder(nn.Module):
    r"""Companion decoder model for the `TabNet` model (which can be
    considered an encoder itself)

    This class is designed to be used with the `EncoderDecoderTrainer` when
    using self-supervised pre-training (see the corresponding section in the
    docs). This class will receive the output from the `TabNet` encoder
    (i.e. the output from the so called 'steps') and '_reconstruct_' the
    embeddings.

    Parameters
    ----------
    embed_dim: int
        Size of the embeddings tensor to be reconstructed.
    n_steps: int, default = 3
        number of decision steps. For a better understanding of the function
        of `n_steps` and the upcoming parameters, please see the
        [paper](https://arxiv.org/abs/1908.07442).
    step_dim: int, default = 8
        Step's output dimension. This is the output dimension that
        `WideDeep` will collect and connect to the output neuron(s).
    dropout: float, default = 0.0
        GLU block's internal dropout
    n_glu_step_dependent: int, default = 2
        number of GLU Blocks (`[FC -> BN -> GLU]`) that are step dependent
    n_glu_shared: int, default = 2
        number of GLU Blocks (`[FC -> BN -> GLU]`) that will be shared
        across decision steps
    ghost_bn: bool, default=True
        Boolean indicating if [Ghost Batch Normalization](https://arxiv.org/abs/1705.08741)
        will be used.
    virtual_batch_size: int, default = 128
        Batch size when using Ghost Batch Normalization
    momentum: float, default = 0.02
        Ghost Batch Normalization's momentum. The dreamquark-ai advises for
        very low values. However high values are used in the original
        publication. During our tests higher values lead to better results

    Attributes
    ----------
    decoder: nn.Module
        decoder that will receive the output from the encoder's steps and will
        reconstruct the embeddings

    Examples
    --------
    >>> import torch
    >>> from pytorch_widedeep.models import TabNetDecoder
    >>> x_inp = [torch.rand(3, 8), torch.rand(3, 8), torch.rand(3, 8)]
    >>> decoder = TabNetDecoder(embed_dim=32, ghost_bn=False)
    >>> res = decoder(x_inp)
    >>> res.shape
    torch.Size([3, 32])
    """

    def __init__(
        self,
        embed_dim: int,
        n_steps: int = 3,
        step_dim: int = 8,
        dropout: float = 0.0,
        n_glu_step_dependent: int = 2,
        n_glu_shared: int = 2,
        ghost_bn: bool = True,
        virtual_batch_size: int = 128,
        momentum: float = 0.02,
    ):
        super(TabNetDecoder, self).__init__()

        self.n_steps = n_steps
        self.step_dim = step_dim
        self.dropout = dropout
        self.n_glu_step_dependent = n_glu_step_dependent
        self.n_glu_shared = n_glu_shared
        self.ghost_bn = ghost_bn
        self.virtual_batch_size = virtual_batch_size
        self.momentum = momentum

        shared_layers = nn.ModuleList()
        for i in range(n_glu_shared):
            if i == 0:
                shared_layers.append(nn.Linear(step_dim, 2 * step_dim, bias=False))
            else:
                shared_layers.append(nn.Linear(step_dim, 2 * step_dim, bias=False))

        self.decoder = nn.ModuleList()
        for step in range(n_steps):
            transformer = FeatTransformer(
                step_dim,
                step_dim,
                dropout,
                shared_layers,
                n_glu_step_dependent,
                ghost_bn,
                virtual_batch_size,
                momentum=momentum,
            )
            self.decoder.append(transformer)

        self.reconstruction_layer = nn.Linear(step_dim, embed_dim, bias=False)
        initialize_non_glu(self.reconstruction_layer, step_dim, embed_dim)

    def forward(self, X: List[Tensor]) -> Tensor:
        out = torch.tensor(0.0)
        for i, x in enumerate(X):
            x = self.decoder[i](x)
            out = torch.add(out, x)
        out = self.reconstruction_layer(out)
        return out
