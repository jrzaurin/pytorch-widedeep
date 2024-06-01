import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_widedeep.wdtypes import List, Tuple, Tensor, Literal, Optional

use_cuda = torch.cuda.is_available()


class MultiTargetRegressionLoss(nn.Module):
    """
    This class is a wrapper around the Pytorch MSELoss. It allows for multi-target
    regression problems. The user can provide a list of weights to apply to each
    target. The loss can be either the sum or the mean of the individual losses

    Parameters
    ----------
    weights: Optional[List[float], default = None]
        List of weights to apply to each target. The length of the list must match
        the number of targets
    reduction: Literal["mean", "sum"], default = "mean
        Specifies the reduction to apply to the output: 'mean' | 'sum'.
        Note that this is NOT the same as the reduction in the MSELoss. This
        reduction is applied after the loss for each target has been computed.

    Examples
    --------
    """

    def __init__(
        self,
        weights: Optional[List[float]] = None,
        reduction: Literal["mean", "sum"] = "mean",
    ):
        super(MultiTargetRegressionLoss, self).__init__()

        self.weights = weights

        if reduction not in ["mean", "sum"]:
            raise ValueError("reduction must be either 'mean' or 'sum'")
        self.reduction = reduction

    def forward(self, input: Tensor, target: Tensor) -> Tensor:

        assert input.size() == target.size()

        if self.weights is not None:
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
        List of integers or tuples with two elements: the index of the target and the
        positive weight for binary classification
    multiclass_config: Optional[List[int | Tuple[int, int] | Tuple[int, List[float]]], default = None
        List of integers or tuples with two elements: the index of the target and the
        number of classes for multiclass classification, or a tuple with the index of
        the target and a list of weights to apply to each class (i.e. the 'weight'
        parameter in the cross_entropy loss)
    weights: Optional[List[float], default = None]
        List of weights to apply to each target. The length of the list must match
        the number of targets
    reduction: Literal["mean", "sum"], default = "sum
        Specifies the reduction to apply to the output: 'mean' | 'sum'.
        Note that this is NOT the same as the reduction in the cross_entropy loss or the
        binary_cross_entropy_with_logits. This reduction is applied after the loss for
        each target has been computed.
    binary_trick: bool, default = False
        If True, each target will be considered independently and the loss
        will be computed as binary_cross_entropy_with_logits. This is a fast
        implementation and it is useful when the targets are not mutually
        exclusive. Note that the 'weights' parameter is not compatible with
        binary_trick=True. Also note that if binary_trick=True,
        the 'binary_config' must be a list of integers and
        the 'multiclass_config' must be a list of tuples with two integers:
        the index of the target and the number of classes.

        **Note**: When using the binary_trick, the binary targets are
          considered as 2 classes. Therefore, the pred_dim parametere of the
          WideDeep class should be adjusted accordingly (adding 2 to per
          binary target). For example, in a problem with a binary target and
          a 4 class multiclassification target, the pred_dim should be 6.


    Examples
    --------
    """

    def __init__(  # noqa: C901
        self,
        binary_config: Optional[List[int | Tuple[int, float]]] = None,
        multiclass_config: Optional[
            List[int | Tuple[int, int] | Tuple[int, List[float]]]
        ] = None,
        weights: Optional[List[float]] = None,
        reduction: Literal["mean", "sum"] = "mean",
        binary_trick: bool = False,
    ):
        super(MultiTargetClassificationLoss, self).__init__()

        assert (
            binary_config is not None or multiclass_config is not None
        ), "Either binary_config or multiclass_config must be provided"

        if reduction not in ["mean", "sum"]:
            raise ValueError("reduction must be either 'mean' or 'sum'")

        self.binary_config = binary_config
        self.multiclass_config = multiclass_config
        self.weights = weights
        self.reduction = reduction
        self.binary_trick = binary_trick

        if self.binary_trick:
            if weights is not None:
                raise ValueError("weights is not compatible with binary_trick=True")

            # TO DO: add support for weights in binary_trick
            if binary_config is not None:
                if any(isinstance(bc, tuple) for bc in binary_config):
                    raise ValueError(
                        "binary_trick=True is only compatible with binary_config as a list of integers"
                    )
                # just to avoid type errors in the forward method
                self._binary_config: List[int] = binary_config  # type: ignore[assignment]
            if multiclass_config is not None:
                if not all(isinstance(mc, tuple) for mc in multiclass_config) and all(
                    isinstance(mc[1], int) for mc in multiclass_config  # type: ignore[index]
                ):
                    raise ValueError(
                        "binary_trick=True is only compatible with multiclass_config as a list of "
                        "tuples with two integers: the index of the target and the number of classes"
                    )
                # just to avoid type errors in the forward method
                self._multiclass_config: List[Tuple[int, int]] = multiclass_config  # type: ignore[assignment]
        else:
            if binary_config is not None:
                self.binary_config_with_pos_weights: List[
                    Tuple[int, Optional[float]]
                ] = []
                for bc in binary_config:
                    if isinstance(bc, tuple):
                        self.binary_config_with_pos_weights.append(bc)
                    else:
                        self.binary_config_with_pos_weights.append((bc, None))

            if multiclass_config is not None:
                self.multiclass_config_with_weights: List[
                    Tuple[int, Optional[List[float]]]
                ] = []
                for mc in multiclass_config:
                    if isinstance(mc, tuple):
                        if isinstance(mc[1], list):
                            self.multiclass_config_with_weights.append((mc[0], mc[1]))
                        else:
                            self.multiclass_config_with_weights.append((mc[0], None))
                    else:
                        self.multiclass_config_with_weights.append((mc, None))

        if weights is not None:
            if len(weights) != (
                len(binary_config) if binary_config is not None else 0
            ) + (len(multiclass_config) if multiclass_config is not None else 0):
                raise ValueError(
                    "The number of weights must match the number of binary and multiclass targets"
                )

    def forward(self, input: Tensor, target: Tensor) -> Tensor:  # noqa: C901

        if self.binary_trick:
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
            loss = F.binary_cross_entropy_with_logits(input, binary_target)
        else:
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
                for idx, mpos_weight in self.multiclass_config_with_weights:
                    # BUG: fix slicing. Need to pass number of classes and slice accordingly
                    _loss = F.cross_entropy(
                        input[:, idx:],
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

            loss = (
                torch.stack(losses).sum()
                if self.reduction == "sum"
                else torch.stack(losses).mean()
            )

        return loss


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
        List of integers or tuples with two elements: the index of the target and the
        positive weight for binary classification
    multiclass_config: Optional[List[int | Tuple[int, int] | Tuple[int, List[float]]], default = None
        List of integers or tuples with two elements: the index of the target and the
        number of classes for multiclass classification, or a tuple with the index of
        the target and a list of weights to apply to each class (i.e. the 'weight'
        parameter in the cross_entropy loss)
    weights: Optional[List[float], default = None]
        List of weights to apply to each target. The length of the list must match
        the number of targets
    reduction: Literal["mean", "sum"], default = "sum
        Specifies the reduction to apply to the output: 'mean' | 'sum'. Note
        that this is NOT the same as the reduction in the cross_entropy loss,
        the binary_cross_entropy_with_logits or the MSELoss. This reduction
        is applied after each target has been computed.
    binary_trick: bool, default = False
        If True, each target for the classification problems will be
        considered independently and the loss will be computed as
        binary_cross_entropy_with_logits. This is a fast implementation and
        it is useful when the targets are not mutually exclusive. Note that
        the 'weights' parameter is not compatible with binary_trick=True.
        Also note that if binary_trick=True, the 'binary_config' must be a
        list of integers and the 'multiclass_config' must be a list of tuples
        with two integers: the index of the target and the number of
        classes.

        **Note**: When using the binary_trick, the binary targets are
          considered as 2 classes. Therefore, the pred_dim parametere of the
          WideDeep class should be adjusted accordingly (adding 2 to per
          binary target). For example, in a problem with a binary target and
          a 4 class multiclassification target, the pred_dim should be 6.

    Examples
    --------

    """

    def __init__(  # noqa: C901
        self,
        regression_config: List[int] = [],
        binary_config: Optional[List[int | Tuple[int, float]]] = None,
        multiclass_config: Optional[
            List[int | Tuple[int, int] | Tuple[int, List[float]]]
        ] = None,
        weights: Optional[List[float]] = None,
        reduction: Literal["mean", "sum"] = "sum",
        binary_trick: bool = False,
    ):

        super(MutilTargetRegressionAndClassificationLoss, self).__init__()

        assert (
            binary_config is not None or multiclass_config is not None
        ), "Either binary_config or multiclass_config must be provided"

        self.regression_config = regression_config

        if weights is not None:
            if len(weights) != (
                len(regression_config)
                + (len(binary_config) if binary_config is not None else 0)
                + (len(multiclass_config) if multiclass_config is not None else 0)
            ):
                raise ValueError(
                    "The number of weights must match the number of regression, binary and multiclass targets"
                )

            self.weights = weights

            self.weights_regression = [
                w for idx, w in enumerate(weights) if idx in regression_config
            ]

            if binary_config is not None:
                binary_idx: List[int] = []
                for bc in binary_config:
                    if isinstance(bc, tuple):
                        binary_idx.append(bc[0])
                    else:
                        binary_idx.append(bc)
                self.weights_binary = [
                    w for idx, w in enumerate(weights) if idx in binary_idx
                ]
            else:
                self.weights_binary = None

            if multiclass_config is not None:
                multiclass_idx: List[int] = []
                for mc in multiclass_config:
                    if isinstance(mc, tuple):
                        multiclass_idx.append(mc[0])
                    else:
                        multiclass_idx.append(mc)
                self.weights_multiclass = [
                    w for idx, w in enumerate(weights) if idx in multiclass_idx
                ]
            else:
                self.weights_multiclass = None
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
            input[:, [self.regression_config]],
            target[:, [self.regression_config]],
        )

        classification_loss = self.multi_target_classification_loss(input, target)

        return regression_loss + classification_loss
