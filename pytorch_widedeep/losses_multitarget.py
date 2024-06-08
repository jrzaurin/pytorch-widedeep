import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_widedeep.wdtypes import (
    List,
    Tuple,
    Union,
    Tensor,
    Literal,
    Optional,
)
from pytorch_widedeep.utils.general_utils import alias

use_cuda = torch.cuda.is_available()


class MultiTargetRegressionLoss(nn.Module):
    """
    This class is a wrapper around the Pytorch MSELoss. It allows for multi-target
    regression problems. The user can provide a list of weights to apply to each
    target. The loss can be either the sum or the mean of the individual losses

    Parameters
    ----------
    weights: Optional[List[float], default = None]
        List of weights to apply to the loss associated to each target. The
        length of the list must match the number of targets.
        Alias: 'target_weights'
    reduction: Literal["mean", "sum"], default = "mean
        Specifies the reduction to apply to the loss associated to each
        target: 'mean' | 'sum'. Note that this is NOT the same as the
        reduction in the MSELoss. This reduction is applied after the loss
        for each target has been computed. Alias: 'target_reduction'

    Examples
    --------
    >>> import torch
    >>> from pytorch_widedeep.losses_multitarget import MultiTargetRegressionLoss
    >>> input = torch.randn(3, 2)
    >>> target = torch.randn(3, 2)
    >>> loss = MultiTargetRegressionLoss(weights=[0.5, 0.5], reduction="mean")
    >>> output = loss(input, target)
    """

    @alias("reduction", ["target_reduction"])
    @alias("weights", ["target_weights"])
    def __init__(
        self,
        weights: Optional[List[float]] = None,
        reduction: Literal["mean", "sum"] = "mean",
    ):
        super(MultiTargetRegressionLoss, self).__init__()

        self.weights = weights
        self.reduction = reduction

        if self.reduction not in ["mean", "sum"]:
            raise ValueError("reduction must be either 'mean' or 'sum'")

    def forward(self, input: Tensor, target: Tensor) -> Tensor:

        assert input.size() == target.size()

        if self.weights is not None:

            assert len(self.weights) == input.size(1), (
                "The number of weights must match the number of targets. "
                f"Got {len(self.weights)} weights and {input.size(1)} targets"
            )

            loss = F.mse_loss(input, target, reduction="none") * torch.tensor(
                self.weights
            ).to(input.device)
        else:
            loss = F.mse_loss(input, target, reduction="none")

        return loss.mean() if self.reduction == "mean" else loss.sum()


class MultiTargetClassificationLoss(nn.Module):
    """
    This class is a wrapper around the Pytorch binary_cross_entropy_with_logits and
    cross_entropy losses. It allows for multi-target classification problems. The
    user can provide a list of weights to apply to each target. The loss can be
    either the sum or the mean of the individual losses

    Parameters
    ----------
    binary_config: Optional[List[int | Tuple[int, float]]], default = None
        List of integers with the index of the target for binary
        classification or tuples with two elements: the index of the targets
        or binary classification and the positive weight for binary
        classification
    multiclass_config: Optional[Tuple[int, int] | Tuple[int, int, List[float]]], default = None
        List of tuples with two or three elements: the index of the target and the
        number of classes for multiclass classification, or a tuple with the index of
        the target, the number of classes and a list of weights to apply to each class
        (i.e. the 'weight' parameter in the cross_entropy loss)
    weights: Optional[List[float], default = None]
        List of weights to apply to the loss associated to each target. The
        length of the list must match the number of targets.
        Alias: 'target_weights'
    reduction: Literal["mean", "sum"], default = "sum
        Specifies the reduction to apply to the loss associated to each
        target: 'mean' | 'sum'. Note that this is NOT the same as the
        reduction in the cross_entropy loss or the
        binary_cross_entropy_with_logits. This reduction is applied after the
        loss for each target has been computed. Alias: 'target_reduction'
    binary_trick: bool, default = False
        If True, each target will be considered independently and the loss
        will be computed as binary_cross_entropy_with_logits. This is a
        faster implementation. Note that the 'weights' parameter is not
        compatible with binary_trick=True. Also note that if
        binary_trick=True, the 'binary_config' must be a list of integers and
        the 'multiclass_config' must be a list of tuples with two integers:
        the index of the target and the number of classes. Finally, if
        binary_trick=True, the binary targets must be the first targets in
        the target tensor.

        :information_source: **NOTE**: When using the binary_trick, the binary targets are
          considered as 2 classes. Therefore, the pred_dim parametere of the
          WideDeep class should be adjusted accordingly (adding 2 to per
          binary target). For example, in a problem with a binary target and
          a 4 class multiclassification target, the pred_dim should be 6.


    Examples
    --------
    >>> import torch
    >>> from pytorch_widedeep.losses_multitarget import MultiTargetClassificationLoss
    >>> input = torch.randn(5, 4)
    >>> input_binary_trick = torch.randn(5, 5)
    >>> target = torch.stack([torch.tensor([0, 1, 0, 1, 1]), torch.tensor([0, 1, 2, 0, 2])], 1)
    >>> loss_1 = MultiTargetClassificationLoss(binary_config=[0], multiclass_config=[(1, 3)], reduction="mean")
    >>> output_1 = loss_1(input, target)
    >>> loss_2 = MultiTargetClassificationLoss(binary_config=[(0, 0.5)], multiclass_config=[(1, 3, [1., 2., 3.])],
    ... reduction="sum", weights=[0.5, 0.5])
    >>> output_2 = loss_2(input, target)
    >>> loss_3 = MultiTargetClassificationLoss(binary_config=[0], multiclass_config=[(1, 3)], binary_trick=True)
    >>> output_3 = loss_3(input_binary_trick, target)
    """

    @alias("reduction", ["target_reduction"])
    @alias("weights", ["target_weights"])
    def __init__(  # noqa: C901
        self,
        binary_config: Optional[List[Union[int, Tuple[int, float]]]] = None,
        multiclass_config: Optional[
            List[Union[Tuple[int, int], Tuple[int, int, List[float]]]]
        ] = None,
        weights: Optional[List[float]] = None,
        reduction: Literal["mean", "sum"] = "mean",
        binary_trick: bool = False,
    ):
        super(MultiTargetClassificationLoss, self).__init__()

        if reduction not in ["mean", "sum"]:
            raise ValueError("reduction must be either 'mean' or 'sum'")

        self.binary_config = binary_config
        self.multiclass_config = multiclass_config
        self.weights = weights
        self.reduction = reduction
        self.binary_trick = binary_trick

        if self.weights is not None:
            if len(self.weights) != (
                len(self.binary_config) if self.binary_config is not None else 0
            ) + (
                len(self.multiclass_config) if self.multiclass_config is not None else 0
            ):
                raise ValueError(
                    "The number of weights must match the number of binary and multiclass targets"
                )

        if self.binary_trick:
            self._check_inputs_with_binary_trick()
            self._binary_config: List[int] = binary_config  # type: ignore[assignment]
            self._multiclass_config: List[Tuple[int, int]] = self.multiclass_config  # type: ignore[assignment]
        else:
            self.binary_config_with_pos_weights = (
                (self._set_binary_config_without_binary_trick())
                if self.binary_config is not None
                else None
            )
            self.multiclass_config_with_weights = (
                (self._set_multiclass_config_without_binary_trick())
                if self.multiclass_config is not None
                else None
            )

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        if self.binary_trick:
            return self._forward_binary_trick(input, target)
        else:
            return self._forward_without_binary_trick(input, target)

    def _forward_binary_trick(self, input: Tensor, target: Tensor) -> Tensor:
        binary_target_tensors: List[Tensor] = []
        if self._binary_config:
            for idx in self._binary_config:
                binary_target_tensors.append(
                    torch.eye(2)[target[:, idx].long()].to(input.device)
                )
        if self._multiclass_config:
            for idx, n_classes in self._multiclass_config:
                binary_target_tensors.append(
                    torch.eye(n_classes)[target[:, idx].long()].to(input.device)
                )
        binary_target = torch.cat(binary_target_tensors, 1)
        return F.binary_cross_entropy_with_logits(input, binary_target)

    def _forward_without_binary_trick(self, input: Tensor, target: Tensor) -> Tensor:
        losses: List[Tensor] = []
        if self.binary_config_with_pos_weights:
            for idx, bpos_weight in self.binary_config_with_pos_weights:
                _loss = F.binary_cross_entropy_with_logits(
                    input[:, idx],
                    target[:, idx].float(),
                    pos_weight=(
                        torch.tensor(bpos_weight).to(input.device)
                        if bpos_weight is not None
                        else None
                    ),
                )
                losses.append(_loss)
        if self.multiclass_config_with_weights:
            for idx, n_classes, mpos_weight in self.multiclass_config_with_weights:
                _loss = F.cross_entropy(
                    input[:, idx : idx + n_classes],
                    target[:, idx].long(),
                    weight=(
                        torch.tensor(mpos_weight).to(input.device)
                        if mpos_weight is not None
                        else None
                    ),
                )
                losses.append(_loss)

            if self.weights is not None:
                losses = [l * w for l, w in zip(losses, self.weights)]  # noqa: E741

        return (
            torch.stack(losses).sum()
            if self.reduction == "sum"
            else torch.stack(losses).mean()
        )

    def _check_inputs_with_binary_trick(self):
        if self.binary_config is not None:
            if any(isinstance(bc, tuple) for bc in self.binary_config):
                raise ValueError(
                    "binary_trick=True is only compatible with binary_config as a list of integers"
                )

        if self.multiclass_config is not None:
            if not all(len(mc) == 2 for mc in self.multiclass_config):
                raise ValueError(
                    "binary_trick=True is only compatible with multiclass_config as a list of "
                    "tuples with two integers: the index of the target and the number of classes"
                )

        if self.binary_config is not None and self.multiclass_config is not None:
            last_binary_idx = (
                self.binary_config[-1][0]
                if isinstance(self.binary_config[-1], tuple)
                else self.binary_config[-1]
            )
            if last_binary_idx >= self.multiclass_config[0][0]:
                raise ValueError(
                    "When using binary_trick=True, the binary targets must be the first targets"
                    " in the target tensor"
                )

    def _set_binary_config_without_binary_trick(
        self,
    ) -> List[Tuple[int, Optional[float]]]:
        binary_config_with_pos_weights: List[Tuple[int, Optional[float]]] = []
        for bc in self.binary_config:
            if isinstance(bc, tuple):
                binary_config_with_pos_weights.append(bc)
            else:
                binary_config_with_pos_weights.append((bc, None))
        return binary_config_with_pos_weights

    def _set_multiclass_config_without_binary_trick(
        self,
    ) -> List[Tuple[int, int, Optional[List[float]]]]:
        multiclass_config_with_weights: List[Tuple[int, int, Optional[List[float]]]] = (
            []
        )
        for mc in self.multiclass_config:
            if len(mc) == 3:
                multiclass_config_with_weights.append(mc)  # type: ignore[arg-type]
            else:
                multiclass_config_with_weights.append((mc[0], mc[1], None))
        return multiclass_config_with_weights


class MutilTargetRegressionAndClassificationLoss(nn.Module):
    """
    This class is a wrapper around the MultiTargetRegressionLoss and the
    MultiTargetClassificationLoss. It allows for multi-target regression and
    classification problems. The user can provide a list of weights to apply to
    each target. The loss can be either the sum or the mean of the individual losses

    Parameters
    ----------
    regression_config: List[int], default = []
        List of integers with the indices of the regression targets
    binary_config: Optional[List[int | Tuple[int, float]]], default = None
        List of integers with the index of the target for binary
        classification or tuples with two elements: the index of the targets
        or binary classification and the positive weight for binary
        classification
    multiclass_config: Optional[Tuple[int, int] | Tuple[int, int, List[float]]], default = None
        List of tuples with two or three elements: the index of the target and the
        number of classes for multiclass classification, or a tuple with the index of
        the target, the number of classes and a list of weights to apply to each class
        (i.e. the 'weight' parameter in the cross_entropy loss)
    weights: Optional[List[float], default = None]
        List of weights to apply to the loss associated to each target. The
        length of the list must match the number of targets.
        Alias: 'target_weights'
    reduction: Literal["mean", "sum"], default = "sum
        Specifies the reduction to apply to the output: 'mean' | 'sum'. Note
        that this is NOT the same as the reduction in the cross_entropy loss,
        the binary_cross_entropy_with_logits or the MSELoss. This reduction
        is applied after each target has been computed. Alias: 'target_reduction'
    binary_trick: bool, default = False
        If True, each target will be considered independently and the loss
        will be computed as binary_cross_entropy_with_logits. This is a
        faster implementation. Note that the 'weights' parameter is not
        compatible with binary_trick=True. Also note that if
        binary_trick=True, the 'binary_config' must be a list of integers and
        the 'multiclass_config' must be a list of tuples with two integers:
        the index of the target and the number of classes. Finally, if
        binary_trick=True, the binary targets must be the first targets in
        the target tensor.

        :information_source: **NOTE**: When using the binary_trick, the binary targets are
          considered as 2 classes. Therefore, the pred_dim parametere of the
          WideDeep class should be adjusted accordingly (adding 2 to per
          binary target). For example, in a problem with a binary target and
          a 4 class multiclassification target, the pred_dim should be 6.

    Examples
    --------
    >>> import torch
    >>> from pytorch_widedeep.losses_multitarget import MutilTargetRegressionAndClassificationLoss
    >>> input = torch.randn(5, 5)
    >>> target = torch.stack([torch.randn(5), torch.tensor([0, 1, 0, 1, 1]), torch.tensor([0, 1, 2, 0, 2])], 1)
    >>> loss = MutilTargetRegressionAndClassificationLoss(regression_config=[0], binary_config=[2],
    ... multiclass_config=[(2, 3)], reduction="mean")
    >>> output = loss(input, target)
    """

    @alias("reduction", ["target_reduction"])
    @alias("weights", ["target_weights"])
    def __init__(  # noqa: C901
        self,
        regression_config: List[int] = [],
        binary_config: Optional[List[Union[int, Tuple[int, float]]]] = None,
        multiclass_config: Optional[
            List[Union[Tuple[int, int], Tuple[int, int, List[float]]]]
        ] = None,
        weights: Optional[List[float]] = None,
        reduction: Literal["mean", "sum"] = "mean",
        binary_trick: bool = False,
    ):

        super(MutilTargetRegressionAndClassificationLoss, self).__init__()

        self.regression_config = regression_config

        assert binary_config is not None or multiclass_config is not None, (
            "Either binary_config or multiclass_config must be provided. "
            "Otherwise, use the MultiTargetRegressionLoss"
        )

        if binary_trick:
            self._check_inputs_with_binary_trick(
                regression_config, binary_config, multiclass_config
            )

        if weights is not None:
            if len(weights) != (
                len(regression_config)
                + (len(binary_config) if binary_config is not None else 0)
                + (len(multiclass_config) if multiclass_config is not None else 0)
            ):
                raise ValueError(
                    "The number of weights must match the number of regression, binary and multiclass targets"
                )

            self.weights_regression = self._prepare_weights_for_regression_targets(
                weights, regression_config
            )
            self.weights_binary = self._prepare_weights_per_binary_targets(
                weights, binary_config
            )
            self.weights_multiclass = self._prepare_weights_per_multiclass_targets(
                weights, multiclass_config
            )
            self.weights = weights
        else:
            self.weights_regression = None
            self.weights_binary = None
            self.weights_multiclass = None

        self.multi_target_regression_loss = MultiTargetRegressionLoss(
            weights=self.weights_regression, reduction=reduction
        )

        self.multi_target_classification_loss = MultiTargetClassificationLoss(
            binary_config=binary_config,
            multiclass_config=multiclass_config,
            weights=(
                self.weights_binary + self.weights_multiclass
                if self.weights_binary is not None
                and self.weights_multiclass is not None
                else (
                    self.weights_binary
                    if self.weights_binary is not None
                    else self.weights_multiclass
                )
            ),
            reduction=reduction,
            binary_trick=binary_trick,
        )

    def forward(self, input: Tensor, target: Tensor) -> Tensor:

        regression_loss = self.multi_target_regression_loss(
            input[:, self.regression_config],
            target[:, self.regression_config],
        )

        if self.multi_target_classification_loss.binary_trick:
            classification_loss = self.multi_target_classification_loss(
                input[:, len(self.regression_config) :], target
            )
        else:
            classification_loss = self.multi_target_classification_loss(input, target)

        return regression_loss + classification_loss

    def _check_inputs_with_binary_trick(
        self,
        regression_config: List[int],
        binary_config: Optional[List[Union[int, Tuple[int, float]]]],
        multiclass_config: Optional[
            List[Union[Tuple[int, int], Tuple[int, int, List[float]]]]
        ],
    ) -> None:

        error_msg = "When using binary_trick=True, the targets order must be: regression, binary and multiclass"

        first_regression_idx = regression_config[0]
        last_regression_idx = regression_config[-1]
        if first_regression_idx != 0:
            raise ValueError(error_msg)

        if binary_config is not None and multiclass_config is not None:
            first_binary_idx = (
                binary_config[0][0]
                if isinstance(binary_config[0], tuple)
                else binary_config[0]
            )
            last_binary_idx = (
                binary_config[-1][0]
                if isinstance(binary_config[-1], tuple)
                else binary_config[-1]
            )
            first_multiclass_idx = multiclass_config[0][0]

            if (first_binary_idx != last_regression_idx + 1) or (
                last_binary_idx >= first_multiclass_idx
            ):
                raise ValueError(error_msg)
        elif binary_config is not None:
            first_binary_idx = (
                binary_config[0][0]
                if isinstance(binary_config[0], tuple)
                else binary_config[0]
            )
            if first_binary_idx != last_regression_idx + 1:
                raise ValueError(error_msg)
        elif multiclass_config is not None:
            first_multiclass_idx = multiclass_config[0][0]
            if first_multiclass_idx != last_regression_idx + 1:
                raise ValueError(error_msg)
        else:
            raise ValueError(
                "Either binary_config or multiclass_config must be provided. "
                "Otherwise, use the MultiTargetRegressionLoss"
            )

    def _prepare_weights_for_regression_targets(
        self,
        weights: List[float],
        regression_config: List[int],
    ) -> List[float]:

        weights_regression = [
            w for idx, w in enumerate(weights) if idx in regression_config
        ]

        return weights_regression

    def _prepare_weights_per_binary_targets(
        self,
        weights: List[float],
        binary_config: Optional[List[Union[int, Tuple[int, float]]]],
    ) -> Optional[List[float]]:

        if binary_config is not None:
            binary_idx: List[int] = []
            for bc in binary_config:
                if isinstance(bc, tuple):
                    binary_idx.append(bc[0])
                else:
                    binary_idx.append(bc)
            weights_binary = [w for idx, w in enumerate(weights) if idx in binary_idx]
        else:
            weights_binary = None

        return weights_binary

    def _prepare_weights_per_multiclass_targets(
        self,
        weights: List[float],
        multiclass_config: Optional[
            List[Union[Tuple[int, int], Tuple[int, int, List[float]]]]
        ],
    ) -> Optional[List[float]]:

        if multiclass_config is not None:
            multiclass_idx: List[int] = [mc[0] for mc in multiclass_config]
            weights_multiclass = [
                w for idx, w in enumerate(weights) if idx in multiclass_idx
            ]
        else:
            weights_multiclass = None

        return weights_multiclass
