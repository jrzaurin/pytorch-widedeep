#


## Alias
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/utils/general_utils.py/#L11)
```python 
Alias(
   primary_name: str, aliases: Union[str, List[str]]
)
```




**Methods:**


### .__init__
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/utils/general_utils.py/#L12)
```python
.__init__(
   primary_name: str, aliases: Union[str, List[str]]
)
```

---
Convert uses of `aliases` to `primary_name` upon calling the decorated
function/method

Parameters
----------
primary_name: String
Preferred name for the parameter, the value of which will be set
to the value of the used alias. If `primary_name` is already
explicitly used on call in addition to any aliases, the value of
`primary_name` will remain unchanged. It only assumes the value of
an alias if the `primary_name` is not used
---
    callable, all `aliases` are removed from its kwargs

Examples
--------

```python

>>> @Alias("a", ["a2"])
... @Alias("b", ["b2"])
... def bar(a, b=None):
...    print(a, b)
>>> foo = Foo(a2="x", b="y")
x y
>>> bar(a2="x", b2="y")
x y
```

### .__call__
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/utils/general_utils.py/#L48)
```python
.__call__(
   wrapped, instance, args, kwargs
)
```


----


### set_default_attr
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/utils/general_utils.py/#L64)
```python
.set_default_attr(
   obj: Any, name: str, value: Any
)
```

---
Set the `name` attribute of `obj` to `value` if the attribute does not
already exist

Parameters
----------
obj: Object
Object whose `name` attribute will be returned (after setting it to
`value`, if necessary)
---
    Name of the attribute to set to `value`, or to return
    exist

Returns
-------
Object
    `obj.name` if it exists. Else, `value`

Examples
--------

```python

>>> set_default_attr(foo, "my_attr", 99)
32
>>> set_default_attr(foo, "other_attr", 9000)
9000
>>> assert foo.my_attr == 32
>>> assert foo.other_attr == 9000
```
