import torch
from torch import nn

from pytorch_widedeep.models import TabNet
from pytorch_widedeep.wdtypes import List, Union, Tensor, Literal, Optional
from pytorch_widedeep.models.tabular.mlp._layers import MLP
from pytorch_widedeep.models._base_wd_model_component import BaseWDModelComponent


class ModelFuser(BaseWDModelComponent):
    """
    This class is a wrapper around a list of models that are associated to the
    different text and/or image columns (and datasets) The class is designed
    to 'fuse' the models using a variety of methods.

    Parameters
    ----------
    models: List[BaseWDModelComponent]
        List of models whose outputs will be fused
    fusion_method: Union[str, List[str]]
        Method to fuse the output of the models. It can be one of
        ['concatenate', 'mean', 'max', 'sum', 'mult', 'dot', 'head'] or a
        list of those, but 'dot'. If a list is provided the output of the
        models will be fused using all the methods in the list and the final
        output will be the concatenation of the outputs of each method
    projection_method: Optional[str]
        If the fusion_method is not 'concatenate', this parameter will
        determine how to project the output of the models to a common
        dimension. It can be one of ['min', 'max', 'mean']. Default is None
    custom_head: Optional[BaseWDModelComponent | nn.Module]
        Custom head to be used to fuse the output of the models. If provided,
        this will take precedence over head_hidden_dims. Also, if
        provided, 'projection_method' will be ignored.
    head_hidden_dims: Optional[List[int]]
        List with the number of neurons per layer in the custom head. If
        custom_head is provided, this parameter will be ignored
    head_activation: Optional[str]
        Activation function to be used in the custom head. Default is None
    head_dropout: Optional[float]
        Dropout to be used in the custom head. Default is None
    head_batchnorm: Optional[bool]
        Whether to use batchnorm in the custom head. Default is None
    head_batchnorm_last: Optional[bool]
        Whether or not batch normalization will be applied to the last of the
        dense layers
    head_linear_first: Optional[bool]
        Boolean indicating the order of the operations in the dense
        layer. If `True: [LIN -> ACT -> BN -> DP]`. If `False: [BN -> DP ->
        LIN -> ACT]`

    Attributes
    ----------
    head: nn.Module or BaseWDModelComponent
        Custom head to be used to fuse the output of the models. If
        custom_head is provided, this will take precedence over
        head_hidden_dims

    Examples
    --------
    >>> from pytorch_widedeep.preprocessing import TextPreprocessor
    >>> from pytorch_widedeep.models import BasicRNN, ModelFuser
    >>> import torch
    >>> import pandas as pd
    >>>
    >>> df = pd.DataFrame({'text_col1': ['hello world', 'this is a test'],
    ... 'text_col2': ['goodbye world', 'this is another test']})
    >>> text_preprocessor_1 = TextPreprocessor(
    ...     text_col="text_col1",
    ...     max_vocab=10,
    ...     min_freq=1,
    ...     maxlen=5,
    ...     n_cpus=1,
    ...     verbose=0)
    >>> text_preprocessor_2 = TextPreprocessor(
    ...     text_col="text_col2",
    ...     max_vocab=10,
    ...     min_freq=1,
    ...     maxlen=5,
    ...     n_cpus=1,
    ...     verbose=0)
    >>> X_text1 = text_preprocessor_1.fit_transform(df)
    >>> X_text2 = text_preprocessor_2.fit_transform(df)
    >>> X_text1_tnsr = torch.from_numpy(X_text1)
    >>> X_text2_tnsr = torch.from_numpy(X_text2)
    >>> rnn1 = BasicRNN(
    ...     vocab_size=len(text_preprocessor_1.vocab.itos),
    ...     embed_dim=4,
    ...     hidden_dim=4,
    ...     n_layers=1,
    ...     bidirectional=False)
    >>> rnn2 = BasicRNN(
    ...     vocab_size=len(text_preprocessor_2.vocab.itos),
    ...     embed_dim=4,
    ...     hidden_dim=4,
    ...     n_layers=1,
    ...     bidirectional=False)
    >>> fused_model = ModelFuser(models=[rnn1, rnn2], fusion_method='concatenate')
    >>> out = fused_model([X_text1_tnsr, X_text2_tnsr])
    """

    def __init__(
        self,
        models: List[BaseWDModelComponent],
        *,
        fusion_method: Union[
            Literal[
                "concatenate",
                "mean",
                "max",
                "sum",
                "mult",
                "dot",
                "head",
            ],
            List[Literal["concatenate", "mean", "max", "sum", "mult", "head"]],
        ],
        projection_method: Optional[Literal["min", "max", "mean"]] = None,
        custom_head: Optional[Union[BaseWDModelComponent, nn.Module]] = None,
        head_hidden_dims: Optional[List[int]] = None,
        head_activation: Optional[str] = None,
        head_dropout: Optional[float] = None,
        head_batchnorm: Optional[bool] = None,
        head_batchnorm_last: Optional[bool] = None,
        head_linear_first: Optional[bool] = None,
    ) -> None:
        super(ModelFuser, self).__init__()

        self.models = nn.ModuleList(models)
        self.fusion_method = fusion_method
        self.projection_method = projection_method

        self.all_output_dim_equal = all(
            model.output_dim == self.models[0].output_dim for model in self.models
        )

        self.check_input_parameters()

        if self.fusion_method == "head":
            assert (
                head_hidden_dims is not None or custom_head is not None
            ), "When using 'head' as fusion_method, either head_hidden_dims or custom_head must be provided"
            if custom_head is not None:
                # custom_head takes precedence over head_hidden_dims (in case
                # both are provided)
                assert hasattr(
                    custom_head, "output_dim"
                ), "custom_head must have an 'output_dim' property"
                self.head: Union[BaseWDModelComponent, nn.Module] = custom_head
            else:
                assert head_hidden_dims is not None
                self.head_hidden_dims = head_hidden_dims
                self.head_activation = head_activation
                self.head_dropout = head_dropout
                self.head_batchnorm = head_batchnorm
                self.head_batchnorm_last = head_batchnorm_last
                self.head_linear_first = head_linear_first

                self.head = MLP(
                    d_hidden=[sum([model.output_dim for model in self.models])]
                    + self.head_hidden_dims,
                    activation=(
                        "relu" if self.head_activation is None else self.head_activation
                    ),
                    dropout=0.0 if self.head_dropout is None else self.head_dropout,
                    batchnorm=(
                        False if self.head_batchnorm is None else self.head_batchnorm
                    ),
                    batchnorm_last=(
                        False
                        if self.head_batchnorm_last is None
                        else self.head_batchnorm_last
                    ),
                    linear_first=(
                        True
                        if self.head_linear_first is None
                        else self.head_linear_first
                    ),
                )

    def forward(self, X: List[Tensor]) -> Tensor:  # noqa: C901
        if self.fusion_method == "head":
            return self._head_fusion(X)
        elif self.fusion_method == "dot":
            return self._dot_fusion(X)
        else:
            return self._other_fusions(X)

    def _head_fusion(self, X: List[Tensor]) -> Tensor:
        return self.head(torch.cat([model(x) for model, x in zip(self.models, X)], -1))

    def _dot_fusion(self, X: List[Tensor]) -> Tensor:
        assert (
            len(X) == 2
        ), "When using 'dot' as fusion_method, only two models can be fused"
        outputs = [model(x) for model, x in zip(self.models, X)]
        return torch.bmm(outputs[1].unsqueeze(1), outputs[0].unsqueeze(2)).view(-1, 1)

    def _other_fusions(self, X: List[Tensor]) -> Tensor:
        fusion_methods = (
            [self.fusion_method]
            if isinstance(self.fusion_method, str)
            else self.fusion_method
        )
        fused_outputs = [self._apply_fusion_method(fm, X) for fm in fusion_methods]  # type: ignore[attr-defined]
        return (
            fused_outputs[0] if len(fused_outputs) == 1 else torch.cat(fused_outputs, 1)
        )

    def _apply_fusion_method(self, fusion_method: str, X: List[Tensor]) -> Tensor:
        if fusion_method == "concatenate":
            return torch.cat([model(x) for model, x in zip(self.models, X)], 1)

        model_outputs = [model(x) for model, x in zip(self.models, X)]
        projections = self._project(model_outputs)
        stacked = torch.stack(projections, -1)

        if fusion_method == "mean":
            return torch.mean(stacked, -1)
        elif fusion_method == "max":
            return torch.max(stacked, -1)[0]
        elif fusion_method == "min":
            return torch.min(stacked, -1)[0]
        elif fusion_method == "sum":
            return torch.sum(stacked, -1)
        elif fusion_method == "mult":
            return torch.prod(stacked, -1)
        else:
            raise ValueError(f"Unsupported fusion method: {fusion_method}")

    def _project(self, X: List[Tensor]) -> List[Tensor]:
        r"""Projects the output of the models to a common dimension."""

        if self.all_output_dim_equal and self.projection_method is None:
            return X

        output_dims = [model.output_dim for model in self.models]

        if self.projection_method == "min":
            proj_dim = min(output_dims)
            idx = output_dims.index(proj_dim)
        elif self.projection_method == "max":
            proj_dim = max(output_dims)
            idx = output_dims.index(proj_dim)
        elif self.projection_method == "mean":
            proj_dim = int(sum(output_dims) / len(output_dims))
            idx = None
        else:
            raise ValueError("projection_method must be one of ['min', 'max', 'mean']")

        x_proj: List[Tensor] = []
        for i, x in enumerate(X):
            if i == idx:
                x_proj.append(x)
            else:
                x_proj.append(
                    nn.Linear(output_dims[i], proj_dim, bias=False, device=x.device)(x)
                )

        return x_proj

    @property
    def output_dim(self) -> int:
        r"""Returns the output dimension of the model."""
        if self.fusion_method == "head":
            output_dim = (
                self.head_hidden_dims[-1]
                if hasattr(self, "head_hidden_dims")
                else self.head.output_dim
            )
        elif self.fusion_method == "dot":
            output_dim = 1
        else:
            output_dim = 0
            if isinstance(self.fusion_method, str):
                fusion_methods = [self.fusion_method]
            else:
                fusion_methods = self.fusion_method  # type: ignore
            for fm in fusion_methods:
                if fm == "concatenate":
                    output_dim += sum([model.output_dim for model in self.models])
                elif self.projection_method == "mean":
                    output_dim += int(
                        sum([model.output_dim for model in self.models])
                        / len(self.models)
                    )
                elif self.projection_method == "min":
                    output_dim += min([model.output_dim for model in self.models])
                elif self.projection_method == "max":
                    output_dim += max([model.output_dim for model in self.models])
                elif self.all_output_dim_equal:
                    output_dim += self.models[0].output_dim
                else:
                    raise ValueError(
                        "projection_method must be one of ['min', 'max', 'mean']"
                    )

        return output_dim

    def check_input_parameters(self):
        self._validate_tabnet()
        self._validate_fusion_methods()
        self._validate_projection_requirements()
        self._validate_head_dot_exclusivity()

    def _validate_tabnet(self):
        if any(isinstance(model, TabNet) for model in self.models):
            raise ValueError(
                "TabNet is not supported in ModelFuser. "
                "Please, use another model for tabular data"
            )

    def _validate_fusion_methods(self):
        valid_methods = [
            "concatenate",
            "min",
            "max",
            "mean",
            "sum",
            "mult",
            "dot",
            "head",
        ]

        if isinstance(self.fusion_method, str):
            if self.fusion_method not in valid_methods:
                raise ValueError(
                    "fusion_method must be one of ['concatenate', 'mean', 'max', 'sum', 'mult', 'dot', 'head'] "
                    "or a list of any those but 'dot'"
                )
        else:
            if not all(fm in valid_methods for fm in self.fusion_method):
                raise ValueError(
                    "fusion_method must be one of ['concatenate', 'mean', 'max', 'sum', 'mult', 'dot', 'head'] "
                    "or a list of those but 'dot'"
                )

    def _validate_projection_requirements(self):
        needs_projection = not self.all_output_dim_equal
        has_size_dependent_method = False

        if isinstance(self.fusion_method, str):
            has_size_dependent_method = self.fusion_method in ["min", "max", "mean"]
        else:
            has_size_dependent_method = all(
                fm in ["min", "max", "mean"] for fm in self.fusion_method
            )

        if has_size_dependent_method and needs_projection:
            if self.projection_method is None:
                raise ValueError(
                    "If 'fusion_method' is not 'concatenate' or 'head', "
                    "and the output dimensions of the models are not equal, "
                    "'projection_method' must be provided"
                )
            elif self.projection_method not in ["min", "max", "mean"]:
                raise ValueError(
                    "projection_method must be one of ['min', 'max', 'mean']"
                )

    def _validate_head_dot_exclusivity(self):
        if isinstance(self.fusion_method, list):
            if any(method in ["head", "dot"] for method in self.fusion_method):
                raise ValueError(
                    "When using 'head' or 'dot' as fusion_method, no other method should be provided"
                )

    def __repr__(self):
        if self.projection_method is not None:
            proj = f"{self.projection_method}"
        else:
            proj = ""

        return f"Fusion method: {self.fusion_method}. Projection method: {proj}\nFused Models:\n{self.models}"
