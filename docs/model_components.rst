The ``models`` module
======================

This module contains the four main components that will comprise a Wide and
Deep model, and the ``WideDeep`` "constructor" class. These four components
are: ``wide``, ``deeptabular``, ``deeptext``, ``deepimage``.

.. note:: ``TabMlp``, ``TabResnet``, ``TabNet``, ``TabTransformer`` and ``SAINT`` can
	all be used as the ``deeptabular``  component of the model and simply
	represent different alternatives

.. autoclass:: pytorch_widedeep.models.wide.Wide
	:exclude-members: forward
	:members:

.. autoclass:: pytorch_widedeep.models.tab_mlp.TabMlp
	:exclude-members: forward
	:members:

.. autoclass:: pytorch_widedeep.models.tab_resnet.TabResnet
	:exclude-members: forward
	:members:

.. autoclass:: pytorch_widedeep.models.tabnet.tab_net.TabNet
	:exclude-members: forward
	:members:

.. autoclass:: pytorch_widedeep.models.transformers.tab_transformer.TabTransformer
	:exclude-members: forward
	:members:

.. autoclass:: pytorch_widedeep.models.transformers.saint.SAINT
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
