import numpy as np
import torch
from tqdm import tqdm
from torch import nn
from torch.utils.data import TensorDataset
from sklearn.model_selection import train_test_split

from pytorch_widedeep.losses import (
    L1Loss,
    MSELoss,
    MSLELoss,
    RMSELoss,
    ZILNLoss,
    FocalLoss,
    HuberLoss,
    RMSLELoss,
    TweedieLoss,
    QuantileLoss,
    FocalR_L1Loss,
    BayesianSELoss,
    FocalR_MSELoss,
    FocalR_RMSELoss,
)
from pytorch_widedeep.wdtypes import (
    Any,
    Dict,
    List,
    Tuple,
    Union,
    Compose,
    Literal,
    Optional,
    Transforms,
)
from pytorch_widedeep.training._wd_dataset import WideDeepDataset
from pytorch_widedeep.training._loss_and_obj_aliases import (
    _LossAliases,
    _ObjectiveToMethod,
)


def tabular_train_val_split(
    seed: int,
    method: str,
    X: np.ndarray,
    y: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    val_split: Optional[float] = None,
) -> Tuple[TensorDataset, Optional[TensorDataset]]:
    r"""
    Function to create the train/val split for the BayesianTrainer where only
    tabular data is used

    Parameters
    ----------
    seed: int
        random seed to be used during train/val split
    method: str
        'regression',  'binary' or 'multiclass'
    X: np.ndarray
        tabular dataset (categorical and continuous features)
    y: np.ndarray
    X_val: np.ndarray, Optional, default = None
        Dict with the validation set, where the keys are the component names
        (e.g: 'wide') and the values the corresponding arrays
    y_val: np.ndarray, Optional, default = None

    Returns
    -------
    train_set: ``TensorDataset``
    eval_set: ``TensorDataset``
    """

    if X_val is not None:
        assert (
            y_val is not None
        ), "if X_val is not None the validation target 'y_val' must also be specified"

        train_set = TensorDataset(
            torch.from_numpy(X),
            torch.from_numpy(y),
        )
        eval_set = TensorDataset(
            torch.from_numpy(X_val),
            torch.from_numpy(y_val),
        )
    elif val_split is not None:
        y_tr, y_val, idx_tr, idx_val = train_test_split(  # type: ignore
            y,
            np.arange(len(y)),
            test_size=val_split,
            random_state=seed,
            stratify=y if method != "regression" else None,
        )
        X_tr, X_val = X[idx_tr], X[idx_val]

        train_set = TensorDataset(
            torch.from_numpy(X_tr),
            torch.from_numpy(y_tr),
        )
        eval_set = TensorDataset(
            torch.from_numpy(X_val),
            torch.from_numpy(y_val),
        )
    else:
        train_set = TensorDataset(
            torch.from_numpy(X),
            torch.from_numpy(y),
        )
        eval_set = None

    return train_set, eval_set


def wd_train_val_split(  # noqa: C901
    seed: int,
    method: Literal["regression", "binary", "multiclass", "qregression", "multitarget"],
    X_wide: Optional[np.ndarray] = None,
    X_tab: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
    X_text: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
    X_img: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
    X_train: Optional[Dict[str, Union[np.ndarray, List[np.ndarray]]]] = None,
    X_val: Optional[Dict[str, Union[np.ndarray, List[np.ndarray]]]] = None,
    val_split: Optional[float] = None,
    target: Optional[np.ndarray] = None,
    transforms: Optional[Union[Transforms, Compose]] = None,
):
    r"""
    Function to create the train/val split for a wide and deep model

    If a validation set (X_val) is passed to the fit method, or val_split is
    specified, the train/val split will happen internally. A number of options
    are allowed in terms of data inputs. For parameter information, please,
    see the ``Trainer``'s' ``.fit()`` method documentation

    Parameters
    ----------
    seed: int
        random seed to be used during train/val split
    method: str
        'regression',  'binary', 'multiclass', 'qregression' or 'multitarget'
    X_wide: np.ndaaray, Optional, default = None
        wide dataset
    X_tab: np.ndarray or List[np.ndarray], Optional, default = None
        tabular dataset (categorical and continuous features)
    X_img: np.ndarray or List[np.ndarray], Optional, default = None
        image dataset
    X_text: np.ndarray or List[np.ndarray], Optional, default = None
        text dataset
    X_val: Dict, Optional, default = None
        Dict with the validation set, where the keys are the component names
        (e.g: 'wide') and the values the corresponding arrays
    val_split: float, Optional, default = None
        Alternatively, the validation split can be specified via a float
    target: np.ndarray, Optional, default = None
    transforms: List, Optional, default = None
        List of Transforms to be applied to the image dataset

    Returns
    -------
    train_set: ``WideDeepDataset``
    eval_set: ``WideDeepDataset``
    """

    if X_val is not None:
        assert X_train is not None and target is None, (
            "if the validation set is passed as a dictionary, the training set must also be a dictionary,"
            " that includes the target"
        )
        train_set = WideDeepDataset(**X_train, transforms=transforms)  # type: ignore
        eval_set = WideDeepDataset(**X_val, transforms=transforms)  # type: ignore
    elif val_split is not None:
        if not X_train:
            assert (
                target is not None
            ), "if the validation split is specified, the target must also be specified"
            X_train = _build_train_dict(X_wide, X_tab, X_text, X_img, target)

        y_tr, y_val, idx_tr, idx_val = train_test_split(
            X_train["target"],
            np.arange(len(X_train["target"])),
            test_size=val_split,
            random_state=seed,
            stratify=(
                X_train["target"]
                if method not in ["regression", "qregression", "multitarget"]
                else None
            ),
        )
        X_tr, X_val = {"target": y_tr}, {"target": y_val}
        if "X_wide" in X_train.keys():
            # the wide component will never be a list, but can still be passed
            # to '_wd_train_val_split_component'
            X_tr, X_val = _wd_train_val_split_component(
                X_train, X_tr, X_val, idx_tr, idx_val, "X_wide"
            )
        if "X_tab" in X_train.keys():
            X_tr, X_val = _wd_train_val_split_component(
                X_train, X_tr, X_val, idx_tr, idx_val, "X_tab"
            )
        if "X_text" in X_train.keys():
            X_tr, X_val = _wd_train_val_split_component(
                X_train, X_tr, X_val, idx_tr, idx_val, "X_text"
            )
        if "X_img" in X_train.keys():
            X_tr, X_val = _wd_train_val_split_component(
                X_train, X_tr, X_val, idx_tr, idx_val, "X_img"
            )
        train_set = WideDeepDataset(**X_tr, transforms=transforms)  # type: ignore
        eval_set = WideDeepDataset(**X_val, transforms=transforms)  # type: ignore
    else:
        if not X_train:
            assert target is not None
            X_train = _build_train_dict(X_wide, X_tab, X_text, X_img, target)
        train_set = WideDeepDataset(**X_train, transforms=transforms)  # type: ignore
        eval_set = None

    return train_set, eval_set


def _wd_train_val_split_component(
    X: Dict[str, Union[np.ndarray, List[np.ndarray]]],
    X_tr: Dict[str, Union[np.ndarray, List[np.ndarray]]],
    X_val: Dict[str, Union[np.ndarray, List[np.ndarray]]],
    idx_tr: Any,  # is a numpy array but sklearn's train_test_split returns a non-sensical type
    idx_val: Any,
    component_type: Literal["X_wide", "X_tab", "X_text", "X_img"],
) -> Tuple[
    Dict[str, Union[np.ndarray, List[np.ndarray]]],
    Dict[str, Union[np.ndarray, List[np.ndarray]]],
]:
    if isinstance(X[component_type], list):
        X_tr[component_type], X_val[component_type] = (
            [X[component_type][i][idx_tr] for i in range(len(X[component_type]))],
            [X[component_type][i][idx_val] for i in range(len(X[component_type]))],
        )
    else:
        X_tr[component_type], X_val[component_type] = (
            X[component_type][idx_tr],
            X[component_type][idx_val],
        )

    return X_tr, X_val


def _build_train_dict(
    X_wide: Optional[np.ndarray],
    X_tab: Optional[Union[np.ndarray, List[np.ndarray]]],
    X_text: Optional[Union[np.ndarray, List[np.ndarray]]],
    X_img: Optional[Union[np.ndarray, List[np.ndarray]]],
    target: np.ndarray,
) -> Dict[str, Union[np.ndarray, List[np.ndarray]]]:
    X_train: Dict[str, Union[np.ndarray, List[np.ndarray]]] = {"target": target}
    if X_wide is not None:
        X_train["X_wide"] = X_wide
    if X_tab is not None:
        X_train["X_tab"] = X_tab
    if X_text is not None:
        X_train["X_text"] = X_text
    if X_img is not None:
        X_train["X_img"] = X_img
    return X_train


def print_loss_and_metric(pb: tqdm, loss: float, score: Optional[Dict] = None):
    r"""
    Function to improve readability and avoid code repetition in the
    training/validation loop within the Trainer's fit method

    Parameters
    ----------
    pb: tqdm
        tqdm object defined as trange(...)
    loss: float
        Loss value
    score: Dict
        Dictionary where the keys are the metric names and the values are the
        corresponding values
    """
    if score is not None:
        pb.set_postfix(
            metrics={
                k: np.round(v.astype(float), 4).tolist() for k, v in score.items()
            },
            loss=loss,
        )
    else:
        pb.set_postfix(loss=loss)


def save_epoch_logs(epoch_logs: Dict, loss: float, score: Optional[Dict], stage: str):
    """
    Function to improve readability and avoid code repetition in the
    training/validation loop within the Trainer's fit method

    Parameters
    ----------
    epoch_logs: Dict
        Dict containing the epoch logs
    loss: float
        loss value
    score: Dict
        Dictionary where the keys are the metric names and the values are the
        corresponding values
    stage: str
        one of 'train' or 'val'
    """
    epoch_logs["_".join([stage, "loss"])] = loss
    if score is not None:
        for k, v in score.items():
            log_k = "_".join([stage, k])
            epoch_logs[log_k] = v
    return epoch_logs


def bayesian_alias_to_loss(loss_fn: str, **kwargs) -> nn.Module:
    r"""Function that returns the corresponding loss function given an alias.
    To be used with the ``BayesianTrainer``

    Parameters
    ----------
    loss_fn: str
        Loss name

    Returns
    -------
    Object
        loss function

    Examples
    --------
    >>> from pytorch_widedeep.training._trainer_utils import bayesian_alias_to_loss
    >>> loss_fn = bayesian_alias_to_loss(loss_fn="binary", weight=None)
    """
    if loss_fn == "binary":
        return nn.BCEWithLogitsLoss(pos_weight=kwargs["weight"], reduction="sum")
    elif loss_fn == "multiclass":
        return nn.CrossEntropyLoss(weight=kwargs["weight"], reduction="sum")
    elif loss_fn == "regression":
        return BayesianSELoss()
        # return BayesianRegressionLoss(noise_tolerance=kwargs["noise_tolerance"])
    else:
        raise ValueError(
            "objective or loss function is not supported. Please consider passing a callable "
            "directly to the Trainer (see docs) or use one of 'binary', 'multiclass', "
            "or 'regression'"
        )


def alias_to_loss(loss_fn: str, **kwargs) -> nn.Module:  # noqa: C901
    r"""Function that returns the corresponding loss function given an alias

    Parameters
    ----------
    loss_fn: str
        Loss name or alias

    Returns
    -------
    Object
        loss function

    Examples
    --------
    >>> from pytorch_widedeep.training._trainer_utils import alias_to_loss
    >>> loss_fn = alias_to_loss(loss_fn="binary_logloss", weight=None)
    """

    if loss_fn in _LossAliases.get("binary"):
        return nn.BCEWithLogitsLoss(pos_weight=kwargs["weight"])
    elif loss_fn in _LossAliases.get("multiclass"):
        return nn.CrossEntropyLoss(weight=kwargs["weight"])
    elif loss_fn in _LossAliases.get("regression"):
        return MSELoss()
    elif loss_fn in _LossAliases.get("mean_absolute_error"):
        return L1Loss()
    elif loss_fn in _LossAliases.get("mean_squared_log_error"):
        return MSLELoss()
    elif loss_fn in _LossAliases.get("root_mean_squared_error"):
        return RMSELoss()
    elif loss_fn in _LossAliases.get("root_mean_squared_log_error"):
        return RMSLELoss()
    elif loss_fn in _LossAliases.get("zero_inflated_lognormal"):
        return ZILNLoss()
    elif loss_fn in _LossAliases.get("quantile"):
        return QuantileLoss()
    elif loss_fn in _LossAliases.get("tweedie"):
        return TweedieLoss()
    elif loss_fn in _LossAliases.get("huber"):
        return HuberLoss()
    elif loss_fn in _LossAliases.get("focalr_l1"):
        return FocalR_L1Loss()
    elif loss_fn in _LossAliases.get("focalr_mse"):
        return FocalR_MSELoss()
    elif loss_fn in _LossAliases.get("focalr_rmse"):
        return FocalR_RMSELoss()
    elif "focal_loss" in loss_fn:
        return FocalLoss(**kwargs)
    else:
        raise ValueError(
            "objective or loss function is not supported. Please consider passing a callable "
            "directly to the Trainer (see docs) or use one of the supported objectives "
            "or loss functions: {}".format(", ".join(_ObjectiveToMethod.keys()))
        )
