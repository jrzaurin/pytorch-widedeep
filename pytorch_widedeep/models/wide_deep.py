import warnings

import torch
import torch.nn as nn

from pytorch_widedeep.wdtypes import *  # noqa: F403
from pytorch_widedeep.models.tab_mlp import MLP

warnings.filterwarnings("default", category=DeprecationWarning)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


class WideDeep(nn.Module):
    def __init__(
        self,
        wide: Optional[nn.Module] = None,
        deeptabular: Optional[nn.Module] = None,
        deeptext: Optional[nn.Module] = None,
        deepimage: Optional[nn.Module] = None,
        deephead: Optional[nn.Module] = None,
        head_hidden_dims: Optional[List[int]] = None,
        head_activation: str = "relu",
        head_dropout: float = 0.1,
        head_batchnorm: bool = False,
        head_batchnorm_last: bool = False,
        head_linear_first: bool = False,
        pred_dim: int = 1,
    ):
        r"""Main collector class that combines all ``wide``, ``deeptabular``
        (which can be a number of architectures), ``deeptext`` and ``deepimage`` models.

        There are two options to combine these models that correspond to the
        two main architectures (there is a higher number of
        "sub-architectures") that ``pytorch-widedeep`` can build.

            - Directly connecting the output of the model components to an ouput neuron(s).

            - Adding a `Fully-Connected Head` (FC-Head) on top of the deep models.
              This FC-Head will combine the output form the ``deeptabular``, ``deeptext`` and
              ``deepimage`` and will be then connected to the output neuron(s).

        Parameters
        ----------
        wide: ``nn.Module``, Optional, default = None
            ``Wide`` model. I recommend using the ``Wide`` class in this
            package. However, it is possible to use a custom model as long as
            is consistent with the required architecture, see
            :class:`pytorch_widedeep.models.wide.Wide`
        deeptabular: ``nn.Module``, Optional, default = None

            currently ``pytorch-widedeep`` implements three possible
            architectures for the `deeptabular` component. These are:
            ``TabMlp``, ``TabResnet`` and ``TabTransformer``.

            1. ``TabMlp`` is simply an embedding layer encoding the categorical
            features that are then concatenated and passed through a series of
            dense (hidden) layers (i.e. and MLP).
            See: ``pytorch_widedeep.models.deep_dense.TabMlp``

            2. ``TabResnet`` is an embedding layer encoding the categorical
            features that are then concatenated and passed through a series of
            ResNet blocks formed by dense layers.
            See ``pytorch_widedeep.models.deep_dense_resnet.TabResnet``

            3. ``TabTransformer`` is detailed in `TabTransformer: Tabular Data
            Modeling Using Contextual Embeddings
            <https://arxiv.org/pdf/2012.06678.pdf>`_. See
            ``pytorch_widedeep.models.tab_transformer.TabTransformer``

            I recommend using on of these as ``deeptabular``. However, a
            custom model as long as is  consistent with the required
            architecture. See
            :class:`pytorch_widedeep.models.deep_dense.TabTransformer`.

        deeptext: ``nn.Module``, Optional, default = None
            Model for the text input. Must be an object of class ``DeepText``
            or a custom model as long as is consistent with the required
            architecture. See
            :class:`pytorch_widedeep.models.deep_dense.DeepText`
        deepimage: ``nn.Module``, Optional, default = None
            Model for the images input. Must be an object of class
            ``DeepImage`` or a custom model as long as is consistent with the
            required architecture. See
            :class:`pytorch_widedeep.models.deep_dense.DeepImage`
        deephead: ``nn.Module``, Optional, default = None
            Custom model by the user that will receive the outtput of the deep
            component. Typically a FC-Head (MLP)
        head_hidden_dims: List, Optional, default = None
            Alternatively, the ``head_hidden_dims`` param can be used to
            specify the sizes of the stacked dense layers in the fc-head e.g:
            ``[128, 64]``. Use ``deephead`` or ``head_hidden_dims``, but not
            both.
        head_dropout: float, default = 0.1
            If ``head_hidden_dims`` is not None, dropout between the layers in
            ``head_hidden_dims``
        head_activation: str, default = "relu"
            If ``head_hidden_dims`` is not None, activation function of the
            head layers. One of "relu", gelu" or "leaky_relu"
        head_batchnorm: bool, default = False
            If ``head_hidden_dims`` is not None, specifies if batch
            normalizatin should be included in the head layers
        head_batchnorm_last: bool, default = False
            If ``head_hidden_dims`` is not None, boolean indicating whether or
            not to apply batch normalization to the last of the dense layers
        head_linear_first: bool, default = False
            If ``head_hidden_dims`` is not None, boolean indicating whether
            the order of the operations in the dense layer. If ``True``:
            ``[LIN -> ACT -> BN -> DP]``. If ``False``: ``[BN -> DP -> LIN ->
            ACT]``
        pred_dim: int, default = 1
            Size of the final wide and deep output layer containing the
            predictions. `1` for regression and binary classification or number
            of classes for multiclass classification.

        Examples
        --------

        >>> from pytorch_widedeep.models import TabResnet, DeepImage, DeepText, Wide, WideDeep
        >>> embed_input = [(u, i, j) for u, i, j in zip(["a", "b", "c"][:4], [4] * 3, [8] * 3)]
        >>> column_idx = {k: v for v, k in enumerate(["a", "b", "c"])}
        >>> wide = Wide(10, 1)
        >>> deeptabular = TabResnet(blocks_dims=[8, 4], column_idx=column_idx, embed_input=embed_input)
        >>> deeptext = DeepText(vocab_size=10, embed_dim=4, padding_idx=0)
        >>> deepimage = DeepImage(pretrained=False)
        >>> model = WideDeep(wide=wide, deeptabular=deeptabular, deeptext=deeptext, deepimage=deepimage)


        .. note:: While I recommend using the ``wide`` and ``deeptabular`` components
            within this package when building the corresponding model components,
            it is very likely that the user will want to use custom text and image
            models. That is perfectly possible. Simply, build them and pass them
            as the corresponding parameters. Note that the custom models MUST
            return a last layer of activations (i.e. not the final prediction) so
            that  these activations are collected by ``WideDeep`` and combined
            accordingly. In addition, the models MUST also contain an attribute
            ``output_dim`` with the size of these last layers of activations. See
            for example :class:`pytorch_widedeep.models.tab_mlp.TabMlp`

        """
        super(WideDeep, self).__init__()

        self._check_model_components(
            wide,
            deeptabular,
            deeptext,
            deepimage,
            deephead,
            head_hidden_dims,
            pred_dim,
        )

        # required as attribute just in case we pass a deephead
        self.pred_dim = pred_dim

        # The main 5 components of the wide and deep assemble
        self.wide = wide
        self.deeptabular = deeptabular
        self.deeptext = deeptext
        self.deepimage = deepimage
        self.deephead = deephead

        if self.deephead is None:
            if head_hidden_dims is not None:
                deep_dim = 0
                if self.deeptabular is not None:
                    deep_dim += self.deeptabular.output_dim  # type:ignore
                if self.deeptext is not None:
                    deep_dim += self.deeptext.output_dim  # type:ignore
                if self.deepimage is not None:
                    deep_dim += self.deepimage.output_dim  # type:ignore
                head_hidden_dims = [deep_dim] + head_hidden_dims
                self.deephead = MLP(
                    head_hidden_dims,
                    head_activation,
                    head_dropout,
                    head_batchnorm,
                    head_batchnorm_last,
                    head_linear_first,
                )
                self.deephead.add_module(
                    "head_out", nn.Linear(head_hidden_dims[-1], pred_dim)
                )
            else:
                if self.deeptabular is not None:
                    self.deeptabular = nn.Sequential(
                        self.deeptabular, nn.Linear(self.deeptabular.output_dim, pred_dim)  # type: ignore
                    )
                if self.deeptext is not None:
                    self.deeptext = nn.Sequential(
                        self.deeptext, nn.Linear(self.deeptext.output_dim, pred_dim)  # type: ignore
                    )
                if self.deepimage is not None:
                    self.deepimage = nn.Sequential(
                        self.deepimage, nn.Linear(self.deepimage.output_dim, pred_dim)  # type: ignore
                    )

    def forward(self, X: Dict[str, Tensor]) -> Tensor:  # type: ignore  # noqa: C901

        # Wide output: direct connection to the output neuron(s)
        if self.wide is not None:
            out = self.wide(X["wide"])
        else:
            batch_size = X[list(X.keys())[0]].size(0)
            out = torch.zeros(batch_size, self.pred_dim).to(device)

        # Deep output: either connected directly to the output neuron(s) or
        # passed through a head first
        if self.deephead:
            if self.deeptabular is not None:
                deepside = self.deeptabular(X["deeptabular"])
            else:
                deepside = torch.FloatTensor().to(device)
            if self.deeptext is not None:
                deepside = torch.cat([deepside, self.deeptext(X["deeptext"])], axis=1)  # type: ignore
            if self.deepimage is not None:
                deepside = torch.cat([deepside, self.deepimage(X["deepimage"])], axis=1)  # type: ignore
            deephead_out = self.deephead(deepside)
            deepside_linear = nn.Linear(deephead_out.size(1), self.pred_dim).to(device)
            return out.add_(deepside_linear(deephead_out))
        else:
            if self.deeptabular is not None:
                out.add_(self.deeptabular(X["deeptabular"]))
            if self.deeptext is not None:
                out.add_(self.deeptext(X["deeptext"]))
            if self.deepimage is not None:
                out.add_(self.deepimage(X["deepimage"]))
            return out

    @staticmethod  # noqa: C901
    def _check_model_components(
        wide,
        deeptabular,
        deeptext,
        deepimage,
        deephead,
        head_hidden_dims,
        pred_dim,
    ):

        if wide is not None:
            assert wide.wide_linear.weight.size(1) == pred_dim, (
                "the 'pred_dim' of the wide component ({}) must be equal to the 'pred_dim' "
                "of the deep component and the overall model itself ({})".format(
                    wide.wide_linear.weight.size(1), pred_dim
                )
            )
        if deeptabular is not None and not hasattr(deeptabular, "output_dim"):
            raise AttributeError(
                "deeptabular model must have an 'output_dim' attribute. "
                "See pytorch-widedeep.models.deep_dense.DeepText"
            )
        if deeptext is not None and not hasattr(deeptext, "output_dim"):
            raise AttributeError(
                "deeptext model must have an 'output_dim' attribute. "
                "See pytorch-widedeep.models.deep_dense.DeepText"
            )
        if deepimage is not None and not hasattr(deepimage, "output_dim"):
            raise AttributeError(
                "deepimage model must have an 'output_dim' attribute. "
                "See pytorch-widedeep.models.deep_dense.DeepText"
            )
        if deephead is not None and head_hidden_dims is not None:
            raise ValueError(
                "both 'deephead' and 'head_hidden_dims' are not None. Use one of the other, but not both"
            )
        if (
            head_hidden_dims is not None
            and not deeptabular
            and not deeptext
            and not deepimage
        ):
            raise ValueError(
                "if 'head_hidden_dims' is not None, at least one deep component must be used"
            )
        if deephead is not None:
            deephead_inp_feat = next(deephead.parameters()).size(1)
            output_dim = 0
            if deeptabular is not None:
                output_dim += deeptabular.output_dim
            if deeptext is not None:
                output_dim += deeptext.output_dim
            if deepimage is not None:
                output_dim += deepimage.output_dim
            assert deephead_inp_feat == output_dim, (
                "if a custom 'deephead' is used its input features ({}) must be equal to "
                "the output features of the deep component ({})".format(
                    deephead_inp_feat, output_dim
                )
            )
