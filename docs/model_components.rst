The ``models`` module
======================

This module contains the four main Wide and Deep model component. These are:
``Wide``, ``DeepDense`` or ``DeepDenseResnet``, ``DeepText`` and ``DeepImage``.

.. note:: ``DeepDense`` and ``DeepDenseResnet`` both correspond to what we
    refer as the `"deep dense"` component of the model and simply represent
    two different alternatives

.. autoclass:: pytorch_widedeep.models.wide.Wide
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: pytorch_widedeep.models.deep_dense.DeepDense
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: pytorch_widedeep.models.deep_dense_resnet.DeepDenseResnet
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: pytorch_widedeep.models.deep_text.DeepText
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: pytorch_widedeep.models.deep_image.DeepImage
    :members:
    :undoc-members:
    :show-inheritance:
