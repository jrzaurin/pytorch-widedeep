The ``utils`` module
====================

Initially the intention was for the ``utils`` module to be hidden from the
user. However, there are a series of utilities there that might be useful for
a number of preprocessing tasks. All the classes and functions discussed here
are available directly from the ``utils`` module. For example, the
``LabelEncoder`` within the ``deeptabular_utils`` submodule can be imported as:

.. code-block:: python

	from pytorch_widedeep.utils import LabelEncoder


Objects
-------

.. toctree::

    deeptabular_utils
    image_utils
    fastai_transforms
    text_utils