from collections import defaultdict


class classproperty:
    """In python 3.9 you can just use

    @classmethod
    @property

    Given that we support 3.7, 3.8 as well as 3.9, let's use this hack
    """

    def __init__(self, func):
        self.func = func

    def __get__(self, decorated_self, decorated_cls):
        return self.func(decorated_cls)


class _LossAliases:
    loss_aliases = {
        "binary": ["binary", "logistic", "binary_logloss", "binary_cross_entropy"],
        "multiclass": [
            "multiclass",
            "multi_logloss",
            "cross_entropy",
            "categorical_cross_entropy",
        ],
        "regression": ["regression", "mse", "l2", "mean_squared_error"],
        "mean_absolute_error": ["mean_absolute_error", "mae", "l1"],
        "mean_squared_log_error": ["mean_squared_log_error", "msle"],
        "root_mean_squared_error": ["root_mean_squared_error", "rmse"],
        "root_mean_squared_log_error": ["root_mean_squared_log_error", "rmsle"],
        "zero_inflated_lognormal": ["zero_inflated_lognormal", "ziln"],
        "quantile": ["quantile"],
        "tweedie": ["tweedie"],
    }

    @classproperty
    def alias_to_loss(cls):
        return {
            loss: alias for alias, losses in cls.loss_aliases.items() for loss in losses
        }

    @classmethod
    def get(cls, loss):
        return cls.loss_aliases.get(loss)


class _ObjectiveToMethod:
    objective_to_method = {
        "binary": "binary",
        "logistic": "binary",
        "binary_logloss": "binary",
        "binary_cross_entropy": "binary",
        "binary_focal_loss": "binary",
        "multiclass": "multiclass",
        "multi_logloss": "multiclass",
        "cross_entropy": "multiclass",
        "categorical_cross_entropy": "multiclass",
        "multiclass_focal_loss": "multiclass",
        "regression": "regression",
        "mse": "regression",
        "l2": "regression",
        "mean_squared_error": "regression",
        "mean_absolute_error": "regression",
        "mae": "regression",
        "l1": "regression",
        "mean_squared_log_error": "regression",
        "msle": "regression",
        "root_mean_squared_error": "regression",
        "rmse": "regression",
        "root_mean_squared_log_error": "regression",
        "rmsle": "regression",
        "zero_inflated_lognormal": "regression",
        "ziln": "regression",
        "tweedie": "regression",
        "quantile": "qregression",
    }

    @classproperty
    def method_to_objecive(cls):
        _method_to_objecive = defaultdict(list)
        for obj, method in cls.objective_to_method.items():
            _method_to_objecive[method].append(obj)
        return _method_to_objecive

    @classmethod
    def keys(cls):
        return cls.objective_to_method.keys()

    @classmethod
    def get(cls, obj):
        return cls.objective_to_method.get(obj)
