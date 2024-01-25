from functools import wraps


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
