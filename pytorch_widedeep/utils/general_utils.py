"""
Code for 'Alias' and 'set_default_attr' taken from the one and only Hunter
McGushion and his library:
https://github.com/HunterMcGushion/hyperparameter_hunter
"""
from typing import Any, List, Union

import wrapt


class Alias:
    def __init__(self, primary_name: str, aliases: Union[str, List[str]]):
        r"""Convert uses of `aliases` to `primary_name` upon calling the decorated
        function/method

        Parameters
        ----------
        primary_name: String
            Preferred name for the parameter, the value of which will be set
            to the value of the used alias. If `primary_name` is already
            explicitly used on call in addition to any aliases, the value of
            `primary_name` will remain unchanged. It only assumes the value of
            an alias if the `primary_name` is not used
        aliases: List, string
            One or multiple string aliases for `primary_name`. If
            `primary_name` is not used on call, its value will be set to that
            of a random alias in `aliases`. Before calling the decorated
            callable, all `aliases` are removed from its kwargs

        Examples
        --------
        >>> class Foo():
        ...     @Alias("a", ["a2"])
        ...     def __init__(self, a, b=None):
        ...         print(a, b)
        >>> @Alias("a", ["a2"])
        ... @Alias("b", ["b2"])
        ... def bar(a, b=None):
        ...    print(a, b)
        >>> foo = Foo(a2="x", b="y")
        x y
        >>> bar(a2="x", b2="y")
        x y"""
        self.primary_name = primary_name
        self.aliases = aliases if isinstance(aliases, list) else [aliases]

    @wrapt.decorator
    def __call__(self, wrapped, instance, args, kwargs):
        for alias in set(self.aliases).intersection(kwargs):
            # Only set if no `primary_name` already. Remove `aliases`, leaving only `primary_name`
            kwargs.setdefault(self.primary_name, kwargs.pop(alias))
            # Record aliases used in `instance.__wd_aliases_used` or `wrapped.__wd_aliases_used`
            if instance:
                set_default_attr(instance, "__wd_aliases_used", {})[
                    self.primary_name
                ] = alias
            else:
                set_default_attr(wrapped, "__wd_aliases_used", {})[
                    self.primary_name
                ] = alias
        return wrapped(*args, **kwargs)


def set_default_attr(obj: Any, name: str, value: Any):
    r"""Set the `name` attribute of `obj` to `value` if the attribute does not
    already exist

    Parameters
    ----------
    obj: Object
        Object whose `name` attribute will be returned (after setting it to
        `value`, if necessary)
    name: String
        Name of the attribute to set to `value`, or to return
    value: Object
        Default value to give to `obj.name` if the attribute does not already
        exist

    Returns
    -------
    Object
        `obj.name` if it exists. Else, `value`

    Examples
    --------
    >>> foo = type("Foo", tuple(), {"my_attr": 32})
    >>> set_default_attr(foo, "my_attr", 99)
    32
    >>> set_default_attr(foo, "other_attr", 9000)
    9000
    >>> assert foo.my_attr == 32
    >>> assert foo.other_attr == 9000
    """
    try:
        return getattr(obj, name)
    except AttributeError:
        setattr(obj, name, value)
    return value
