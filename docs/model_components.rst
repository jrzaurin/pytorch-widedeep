The ``models`` module
======================

This module contains the four main models that will comprise Wide and Deep
model, and the ``WideDeep`` "constructor" class. These four components are:
``Wide``, ``DeepDense``, ``DeepDenseResnet``, ``TabTransformer``, ``DeepText``
and ``DeepImage``.

.. note:: ``DeepDense``, ``DeepDenseResnet`` and ``TabTransformer`` correspond to
    what we refer as the ``deeptabular`` component of the model and simply represent
    different alternatives

.. autoclass:: pytorch_widedeep.models.wide.Wide
	:exclude-members: forward
	:members:

.. autoclass:: pytorch_widedeep.models.tab_mlp.TabMlp
	:exclude-members: forward
	:members:

.. autoclass:: pytorch_widedeep.models.tab_resnet.TabResnet
	:exclude-members: forward
	:members:

.. autoclass:: pytorch_widedeep.models.tab_transformer.TabTransformer
	:exclude-members: forward
	:members:

.. autoclass:: pytorch_widedeep.models.deep_text.DeepText
	:exclude-members: forward
	:members:

.. autoclass:: pytorch_widedeep.models.deep_image.DeepImage
	:exclude-members: forward
	:members:

.. autoclass:: pytorch_widedeep.models.wide_deep.WideDeep
	:exclude-members: forward
	:members:
