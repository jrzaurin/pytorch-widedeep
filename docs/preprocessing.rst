The ``preprocessing`` module
============================

This module contains the classes that are used to prepare the data before
being passed to the models. There is one Preprocessor per data mode or
model component: ``wide``, ``deeptabular``, ``deepimage`` and ``deeptext``.

.. autoclass:: pytorch_widedeep.preprocessing.WidePreprocessor
	:members:
	:undoc-members:

.. autoclass:: pytorch_widedeep.preprocessing.TabPreprocessor
	:members:
	:undoc-members:

.. autoclass:: pytorch_widedeep.preprocessing.TextPreprocessor
	:members:
	:undoc-members:

.. autoclass:: pytorch_widedeep.preprocessing.ImagePreprocessor
	:members:
	:undoc-members:
