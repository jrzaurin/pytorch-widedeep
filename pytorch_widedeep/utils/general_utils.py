from functools import wraps

import torch
from torch import Tensor


def setup_device() -> str:
    if torch.cuda.is_available():
        return f"cuda:{torch.cuda.current_device()}"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def to_device(X: Tensor, device: str) -> Tensor:
    # Adjustmet in case the backend is mps which does not support float64
    if device == "mps" and X.dtype == torch.float64:
        X = X.float()
    return X.to(device)


def to_device_model(model, device: str):  # noqa: C901
    # insistent transformation since it some cases overall approaches such as
    # model.to('mps') do not work

    if device == "cpu" or (device.startswith("cuda") and torch.cuda.is_available()):
        return model.to(device)

    if device == "mps":
        try:
            return model.to(device)
        except (RuntimeError, TypeError):

            def convert(t):
                if isinstance(t, torch.Tensor):
                    return t.float().to(device)
                return t

            model.apply(lambda module: module._apply(convert))

            for param in model.parameters():
                if param.device.type != "mps":
                    param.data = param.data.float().to(device)

            for buffer_name, buffer in model.named_buffers():
                if buffer.device.type != "mps":
                    model._buffers[buffer_name] = buffer.float().to(device)

    return model


def alias(original_name: str, alternative_names: list):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for alt_name in alternative_names:
                if alt_name in kwargs:
                    kwargs[original_name] = kwargs.pop(alt_name)
                    break
            return func(*args, **kwargs)

        return wrapper

    return decorator
