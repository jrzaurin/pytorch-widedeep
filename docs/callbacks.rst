Callbacks
=========

Here are the 4 callbacks available in ``pytorch-widedepp``: ``History``,
``LRHistory``, ``ModelCheckpoint`` and ``EarlyStopping``.

.. note:: ``History`` runs by default, so it should not be passed
    to the ``Trainer``

.. autoclass:: pytorch_widedeep.callbacks.History
	:members:

.. autoclass:: pytorch_widedeep.callbacks.LRShedulerCallback
	:members:

.. autoclass:: pytorch_widedeep.callbacks.MetricCallback
	:members:

.. autoclass:: pytorch_widedeep.callbacks.LRHistory
	:members:

.. autoclass:: pytorch_widedeep.callbacks.ModelCheckpoint
	:members:

.. autoclass:: pytorch_widedeep.callbacks.EarlyStopping
	:members:
