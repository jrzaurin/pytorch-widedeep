The ``preprocessing`` module
============================

This module contains the classes that are used to prepare the data before
being passed to the Wide and Deep `constructor` class. There is one
Preprocessor per model component: ``wide``, ``deeptabular``, ``deepimage`` and
``deeptext``.

.. autoclass:: pytorch_widedeep.preprocessing.WidePreprocessor
	:members:
	:undoc-members:
	:show-inheritance:

.. autoclass:: pytorch_widedeep.preprocessing.TabPreprocessor
	:members:
	:undoc-members:
	:show-inheritance:

.. autoclass:: pytorch_widedeep.preprocessing.TextPreprocessor
	:members:
	:undoc-members:
	:show-inheritance:

.. autoclass:: pytorch_widedeep.preprocessing.ImagePreprocessor
	:members:
	:undoc-members:
	:show-inheritance:
