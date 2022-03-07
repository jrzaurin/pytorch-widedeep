"""
During the development of the package I realised that there is a typing
inconsistency. The input components of a Wide and Deep model are of type
nn.Module. These change type internally to nn.Sequential. While nn.Sequential
is an instance of nn.Module the oppossite is, of course, not true. This does
not affect any funcionality of the package, but it is something that needs
fixing. However, while fixing is simple (simply define new attributes that
are the nn.Sequential objects), its implications are quite wide within the
package (involves changing a number of tests and tutorials). Therefore, I
will introduce that fix when I do a major release. For now, we live with it.
"""

import warnings

import torch
import torch.nn as nn

from pytorch_widedeep.wdtypes import *  # noqa: F403
from pytorch_widedeep.models.fds_layer import FDSLayer
from pytorch_widedeep.models._get_activation_fn import get_activation_fn
from pytorch_widedeep.models.tabular.mlp._layers import MLP
from pytorch_widedeep.models.tabular.tabnet.tab_net import TabNetPredLayer

warnings.filterwarnings("default", category=UserWarning)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


class WideDeep(nn.Module):
    r"""Main collector class that combines all ``wide``, ``deeptabular``
    ``deeptext`` and ``deepimage`` models.

    There are two options to combine these models that correspond to the
    two main architectures that ``pytorch-widedeep`` can build.

        - Directly connecting the output of the model components to an ouput neuron(s).

        - Adding a `Fully-Connected Head` (FC-Head) on top of the deep models.
          This FC-Head will combine the output form the ``deeptabular``, ``deeptext`` and
          ``deepimage`` and will be then connected to the output neuron(s).

    Parameters
    ----------
    wide: ``nn.Module``, Optional, default = None
        ``Wide`` model. This is a linear model where the non-linearities are
        captured via crossed-columns.
    deeptabular: ``nn.Module``, Optional, default = None
        Currently this library implements a number of possible architectures
        for the ``deeptabular`` component. See the documenation of the
        package.
    deeptext: ``nn.Module``, Optional, default = None
        Currently this library implements a number of possible architectures
        for the ``deeptext`` component. See the documenation of the
        package.
    deepimage: ``nn.Module``, Optional, default = None
        Currently this library uses ``torchvision`` and implements a number of
        possible architectures for the ``deepimage`` component. See the
        documenation of the package.
    head_hidden_dims: List, Optional, default = None
        List with the sizes of the dense layers in the head e.g: [128, 64]
    head_activation: str, default = "relu"
        Activation function for the dense layers in the head. Currently
        `'tanh'`, `'relu'`, `'leaky_relu'` and `'gelu'` are supported
    head_dropout: float, Optional, default = None
        Dropout of the dense layers in the head
    head_batchnorm: bool, default = False
        Boolean indicating whether or not to include batch normalization in
        the dense layers that form the `'rnn_mlp'`
    head_batchnorm_last: bool, default = False
        Boolean indicating whether or not to apply batch normalization to the
        last of the dense layers in the head
    head_linear_first: bool, default = False
        Boolean indicating whether the order of the operations in the dense
        layer. If ``True: [LIN -> ACT -> BN -> DP]``. If ``False: [BN -> DP ->
        LIN -> ACT]``
    deephead: ``nn.Module``, Optional, default = None
        Alternatively, the user can pass a custom model that will receive the
        output of the deep component. If ``deephead`` is not None all the
        previous fc-head parameters will be ignored
    enforce_positive: bool, default = False
        Boolean indicating if the output from the final layer must be
        positive. This is important if you are using loss functions with
        non-negative input restrictions, e.g. RMSLE, or if you know your
        predictions are bounded in between 0 and inf
    enforce_positive_activation: str, default = "softplus"
        Activation function to enforce that the final layer has a positive
        output. `'softplus'` or `'relu'` are supported.
    pred_dim: int, default = 1
        Size of the final wide and deep output layer containing the
        predictions. `1` for regression and binary classification or number
        of classes for multiclass classification.
    with_fds: bool, default = False
        If feature distribution smoothing (FDS) should be applied before the
        final prediction layer. Only available for regression problems. See
        `Delving into Deep Imbalanced Regression <https://arxiv.org/abs/2102.09554>`_
        for details.
    fds_config: dict, default = None
        Dictionary defining specific values for
        ``FeatureDistributionSmoothing`` layer

    Examples
    --------

    >>> from pytorch_widedeep.models import TabResnet, Vision, BasicRNN, Wide, WideDeep
    >>> embed_input = [(u, i, j) for u, i, j in zip(["a", "b", "c"][:4], [4] * 3, [8] * 3)]
    >>> column_idx = {k: v for v, k in enumerate(["a", "b", "c"])}
    >>> wide = Wide(10, 1)
    >>> deeptabular = TabResnet(blocks_dims=[8, 4], column_idx=column_idx, cat_embed_input=embed_input)
    >>> deeptext = BasicRNN(vocab_size=10, embed_dim=4, padding_idx=0)
    >>> deepimage = Vision()
    >>> model = WideDeep(wide=wide, deeptabular=deeptabular, deeptext=deeptext, deepimage=deepimage)


    .. note:: It is possible to use custom components to build Wide & Deep models.
        Simply, build them and pass them as the corresponding parameters. Note
        that the custom models MUST return a last layer of activations
        (i.e. not the final prediction) so that  these activations are
        collected by ``WideDeep`` and combined accordingly. In addition, the
        models MUST also contain an attribute ``output_dim`` with the size of
        these last layers of activations. See for
        example :class:`pytorch_widedeep.models.tab_mlp.TabMlp`
    """

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
        enforce_positive: bool = False,
        enforce_positive_activation: str = "softplus",
        pred_dim: int = 1,
        with_fds: bool = False,
        **fds_config,
    ):
        super(WideDeep, self).__init__()

        self._check_inputs(
            wide,
            deeptabular,
            deeptext,
            deepimage,
            deephead,
            head_hidden_dims,
            pred_dim,
            with_fds,
        )

        # required as attribute just in case we pass a deephead
        self.pred_dim = pred_dim

        # The main 5 components of the wide and deep assemble
        self.wide = wide
        self.deeptabular = deeptabular
        self.deeptext = deeptext
        self.deepimage = deepimage
        self.deephead = deephead
        self.enforce_positive = enforce_positive
        self.with_fds = with_fds

        if self.deeptabular is not None:
            self.is_tabnet = deeptabular.__class__.__name__ == "TabNet"
        else:
            self.is_tabnet = False

        if self.deephead is None and head_hidden_dims is not None:
            self._build_deephead(
                head_hidden_dims,
                head_activation,
                head_dropout,
                head_batchnorm,
                head_batchnorm_last,
                head_linear_first,
            )
        elif self.deephead is not None:
            pass
        else:
            self._add_pred_layer()

        if self.with_fds:
            self.fds_layer = FDSLayer(feature_dim=self.deeptabular.output_dim, **fds_config)  # type: ignore[arg-type]

        if self.enforce_positive:
            self.enf_pos = get_activation_fn(enforce_positive_activation)

    def forward(
        self,
        X: Dict[str, Tensor],
        y: Optional[Tensor] = None,
        epoch: Optional[int] = None,
    ):

        if self.with_fds:
            return self._forward_deep_with_fds(X, y, epoch)

        wide_out = self._forward_wide(X)
        if self.deephead:
            deep = self._forward_deephead(X, wide_out)
        else:
            deep = self._forward_deep(X, wide_out)

        if self.enforce_positive:
            return self.enf_pos(deep)
        else:
            return deep

    def _build_deephead(
        self,
        head_hidden_dims,
        head_activation,
        head_dropout,
        head_batchnorm,
        head_batchnorm_last,
        head_linear_first,
    ):
        deep_dim = 0
        if self.deeptabular is not None:
            deep_dim += self.deeptabular.output_dim
        if self.deeptext is not None:
            deep_dim += self.deeptext.output_dim
        if self.deepimage is not None:
            deep_dim += self.deepimage.output_dim

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
            "head_out", nn.Linear(head_hidden_dims[-1], self.pred_dim)
        )

    def _add_pred_layer(self):
        # if FDS the FDS Layer already includes the pred layer
        if self.deeptabular is not None and not self.with_fds:
            if self.is_tabnet:
                self.deeptabular = nn.Sequential(
                    self.deeptabular,
                    TabNetPredLayer(self.deeptabular.output_dim, self.pred_dim),
                )
            else:
                self.deeptabular = nn.Sequential(
                    self.deeptabular,
                    nn.Linear(self.deeptabular.output_dim, self.pred_dim),
                )
        if self.deeptext is not None:
            self.deeptext = nn.Sequential(
                self.deeptext, nn.Linear(self.deeptext.output_dim, self.pred_dim)
            )
        if self.deepimage is not None:
            self.deepimage = nn.Sequential(
                self.deepimage, nn.Linear(self.deepimage.output_dim, self.pred_dim)
            )

    def _forward_wide(self, X):
        if self.wide is not None:
            out = self.wide(X["wide"])
        else:
            batch_size = X[list(X.keys())[0]].size(0)
            out = torch.zeros(batch_size, self.pred_dim).to(device)

        return out

    def _forward_deephead(self, X, wide_out):
        if self.deeptabular is not None:
            if self.is_tabnet:
                tab_out = self.deeptabular(X["deeptabular"])
                deepside, M_loss = tab_out[0], tab_out[1]
            else:
                deepside = self.deeptabular(X["deeptabular"])
        else:
            deepside = torch.FloatTensor().to(device)
        if self.deeptext is not None:
            deepside = torch.cat([deepside, self.deeptext(X["deeptext"])], axis=1)
        if self.deepimage is not None:
            deepside = torch.cat([deepside, self.deepimage(X["deepimage"])], axis=1)

        deephead_out = self.deephead(deepside)
        deepside_out = nn.Linear(deephead_out.size(1), self.pred_dim).to(device)

        if self.is_tabnet:
            res = (wide_out.add_(deepside_out(deephead_out)), M_loss)
        else:
            res = wide_out.add_(deepside_out(deephead_out))

        return res

    def _forward_deep(self, X, wide_out):
        if self.deeptabular is not None:
            if self.is_tabnet:
                tab_out, M_loss = self.deeptabular(X["deeptabular"])
                wide_out.add_(tab_out)
            else:
                wide_out.add_(self.deeptabular(X["deeptabular"]))
        if self.deeptext is not None:
            wide_out.add_(self.deeptext(X["deeptext"]))
        if self.deepimage is not None:
            wide_out.add_(self.deepimage(X["deepimage"]))

        if self.is_tabnet:
            res = (wide_out, M_loss)
        else:
            res = wide_out

        return res

    def _forward_deep_with_fds(
        self,
        X: Dict[str, Tensor],
        y: Optional[Tensor] = None,
        epoch: Optional[int] = None,
    ):
        res = self.fds_layer(self.deeptabular(X["deeptabular"]), y, epoch)
        if self.enforce_positive:
            if isinstance(res, Tuple):  # type: ignore[arg-type]
                out = (res[0], self.enf_pos(res[1]))
            else:
                out = self.enf_pos(res)
        else:
            out = res
        return out

    @staticmethod  # noqa: C901
    def _check_inputs(  # noqa: C901
        wide,
        deeptabular,
        deeptext,
        deepimage,
        deephead,
        head_hidden_dims,
        pred_dim,
        with_fds,
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
                "See pytorch-widedeep.models.deep_text.DeepText"
            )
        if deeptabular is not None:
            is_tabnet = deeptabular.__class__.__name__ == "TabNet"
            has_wide_text_or_image = (
                wide is not None or deeptext is not None or deepimage is not None
            )
            if is_tabnet and has_wide_text_or_image:
                warnings.warn(
                    "'WideDeep' is a model comprised by multiple components and the 'deeptabular'"
                    " component is 'TabNet'. We recommend using 'TabNet' in isolation."
                    " The reasons are: i)'TabNet' uses sparse regularization which partially losses"
                    " its purpose when used in combination with other components."
                    " If you still want to use a multiple component model with 'TabNet',"
                    " consider setting 'lambda_sparse' to 0 during training. ii) The feature"
                    " importances will be computed only for TabNet but the model will comprise multiple"
                    " components. Therefore, such importances will partially lose their 'meaning'.",
                    UserWarning,
                )
        if deeptext is not None and not hasattr(deeptext, "output_dim"):
            raise AttributeError(
                "deeptext model must have an 'output_dim' attribute. "
                "See pytorch-widedeep.models.deep_text.DeepText"
            )
        if deepimage is not None and not hasattr(deepimage, "output_dim"):
            raise AttributeError(
                "deepimage model must have an 'output_dim' attribute. "
                "See pytorch-widedeep.models.deep_text.DeepText"
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

        if with_fds and (
            (
                wide is not None
                or deeptext is not None
                or deepimage is not None
                or deephead is not None
            )
            or pred_dim != 1
        ):
            raise ValueError(
                """Feature Distribution Smoothing (FDS) is supported when using only a deeptabular component"
                " and for regression problems."""
            )
