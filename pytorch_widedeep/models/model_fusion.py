import torch
from torch import nn

from pytorch_widedeep.wdtypes import List, Tensor, Literal, Optional
from pytorch_widedeep.models.tabular.mlp._layers import MLP
from pytorch_widedeep.models._base_wd_model_component import (
    BaseWDModelComponent,
)


class ModelFuser(BaseWDModelComponent):
    """

    This class is a wrapper around a list of models that are associated to the
    different text and/or image columns (and datasets) The class is designed
    to 'fuse' the models using a variety of methods.

    Parameters
    ----------
    models: List[BaseWDModelComponent]
        List of models to be fused
    fusion_method: Union[str, List[str]]
        Method to fuse the output of the models. It can be one of
        ['concatenate', 'mean', 'max', 'sum', 'mult', 'head'] or a list of
        those. If a list is provided the output of the models will be fused
        using all the methods in the list and the final output will be the
        concatenation of the output of each method
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
    heda: nn.Module or BaseWDModelComponent
        Custom head to be used to fuse the output of the models. If
        custom_head is provided, this will take precedence over
        head_hidden_dims

    """

    def __init__(
        self,
        models: List[BaseWDModelComponent],
        *,
        fusion_method: (
            Literal[
                "concatenate",
                "mean",
                "max",
                "sum",
                "mult",
                "head",
            ]
            | List[Literal["concatenate", "mean", "max", "sum", "mult", "head"]]
        ),
        projection_method: Optional[Literal["min", "max", "mean"]] = None,
        custom_head: Optional[BaseWDModelComponent | nn.Module] = None,
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
                self.head: BaseWDModelComponent | nn.Module = custom_head
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
            return self.head(
                torch.cat([model(x) for model, x in zip(self.models, X)], -1)
            )
        else:
            if isinstance(self.fusion_method, str):
                fusion_methods = [self.fusion_method]
            else:
                fusion_methods = self.fusion_method

            fused_outputs: List[Tensor] = []
            for fm in fusion_methods:
                if fm == "concatenate":
                    out = torch.cat([model(x) for model, x in zip(self.models, X)], 1)
                else:

                    model_outputs = [model(x) for model, x in zip(self.models, X)]
                    projections = self.project(model_outputs)

                    if fm == "mean":
                        out = torch.mean(torch.stack(projections, -1), -1)
                    elif fm == "max":
                        out, _ = torch.max(torch.stack(projections, -1), -1)
                    elif fm == "min":
                        out, _ = torch.min(torch.stack(projections, -1), -1)
                    elif fm == "sum":
                        out = torch.sum(torch.stack(projections, -1), -1)
                    elif fm == "mult":
                        out = torch.prod(torch.stack(projections, -1), -1)
                    else:
                        # This should never happen, but avoids type errors
                        raise ValueError(
                            "fusion_method must be one of ['concatenate', 'mean', 'max', 'sum', 'mult', 'head'] "
                            "or a list of those"
                        )
                fused_outputs.append(out)

            if len(fused_outputs) == 1:
                return fused_outputs[0]
            else:
                return torch.cat(fused_outputs, 1)

    def project(self, X: List[Tensor]) -> List[Tensor]:
        r"""Projects the output of the models to a common dimension."""

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
                x_proj.append(nn.Linear(output_dims[i], proj_dim, bias=False)(x))

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
        else:
            output_dim = 0
            if isinstance(self.fusion_method, str):
                fusion_methods = [self.fusion_method]
            else:
                fusion_methods = self.fusion_method
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
                else:
                    raise ValueError(
                        "projection_method must be one of ['min', 'max', 'mean']"
                    )

        return output_dim

    def check_input_parameters(self):
        if isinstance(self.fusion_method, str):
            if not any(
                x == self.fusion_method
                for x in ["concatenate", "min", "max", "mean", "sum", "mult", "head"]
            ):
                raise ValueError(
                    "fusion_method must be one of ['concatenate', 'mean', 'max', 'sum', 'mult', 'head'] "
                    "or a list of those"
                )

            if any(x in self.fusion_method for x in ["min", "max", "mean"]):
                assert (
                    self.projection_method is not None
                ), "If 'fusion_method' is not 'concatenate', 'projection_method' must be provided"

        else:
            if not all(
                any(
                    x == fm
                    for x in [
                        "concatenate",
                        "min",
                        "max",
                        "mean",
                        "sum",
                        "mult",
                        "head",
                    ]
                )
                for fm in self.fusion_method
            ):
                raise ValueError(
                    "fusion_method must be one of ['concatenate', 'mean', 'max', 'sum', 'mult', 'head'] "
                    "or a list of those"
                )

            if all(
                any(x in fm for x in ["min", "max", "mean"])
                and self.projection_method is not None
                for fm in self.fusion_method
            ):
                assert (
                    self.projection_method is not None
                ), "If 'fusion_method' is not 'concatenate', 'projection_method' must be provided"

        if (
            not any(x == self.projection_method for x in ["min", "max", "mean"])
            and "head" not in self.fusion_method
        ):
            raise ValueError("projection_method must be one of ['min', 'max', 'mean']")

        if "head" in self.fusion_method and isinstance(self.fusion_method, list):
            raise ValueError(
                "When using 'head' as fusion_method, no other method should be provided"
            )

    def __repr__(self):
        if self.projection_method is not None:
            proj = f"{self.projection_method}"
        else:
            proj = ""

        return f"Fusion method: {self.fusion_method}. Projection method: {proj}\nFused Models:\n{self.models}"
